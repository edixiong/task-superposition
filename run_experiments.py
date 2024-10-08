from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
import yaml
import tiktoken

import datetime
import pytz
import os

import random
import argparse
from yaml import Loader
import tasks
from config import PromptConfig, parse_raw_config_for_prompt

from openai import OpenAI
from copy import deepcopy

OPENAI_API_KEY = "<your-openai-key>"
PRETRAINED_MODELS_DIR = "<path-to-pretrained-models>"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_time() -> str:
    central_tz = pytz.timezone('America/Chicago')

    current_utc = datetime.datetime.now(pytz.utc)
    current_central = current_utc.astimezone(central_tz)

    formatted_time = current_central.strftime('%m-%d-%H-%M-%S')
    return formatted_time

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_prompt(prompt_config: PromptConfig, tokenizer: AutoTokenizer):
    examples = []
    task_funcs = [getattr(tasks, f"generate_task_{prompt_config.task_funcs_dict[i]['name']}") for i in range(prompt_config.num_tasks)]
    task_indices = np.array([999] * prompt_config.num_examples)
    num_tasks = len(task_funcs)
    if prompt_config.order == 'random':
        indices = np.arange(prompt_config.num_examples)
        for i in range(num_tasks):
            indices_select = np.random.choice(indices, prompt_config.task_examples[i], replace=False)
            task_indices[indices_select] = i
            indices = np.setdiff1d(indices, indices_select, assume_unique=True)
    else:
        raise NotImplementedError
    for idx in task_indices:
        task_func = task_funcs[idx]
        task_kwargs = prompt_config.task_funcs_dict[idx]['kwargs']
        example = task_func(examples, tokenizer, **task_kwargs)
        examples.append(example)
    examples_str = "\n".join(examples)

    question_func = getattr(tasks, f"question_{prompt_config.question_name}")
    question_kwargs = prompt_config.question_kwargs
    
    question_str, output_choices = question_func(examples, tokenizer, **question_kwargs)
    prompt = prompt_config.prompt_template.format(examples_str=examples_str, question_str=question_str)
    return prompt, output_choices

def get_probs(device, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, output_choices: List[str]) -> List[float]:
    num_choices = len(output_choices)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_num_tokens = prompt_ids.size()[-1]
    combined_output_lst = [prompt + output for output in output_choices]
    combined_pt_lst = [tokenizer(c, return_tensors="pt").input_ids.to(device) for c in combined_output_lst]
    choices_probs = []
    for input in combined_pt_lst:
        with torch.no_grad():
            logits = model(input).logits.squeeze()[prompt_num_tokens-1:-1]
        expected_out_ids = input.squeeze()[prompt_num_tokens:].unsqueeze(dim=1)
        probs = F.softmax(logits, dim=1)
        pred_probs = torch.gather(probs, dim=1, index=expected_out_ids).squeeze()
        choices_probs.append(torch.prod(pred_probs).item())
    return choices_probs

def get_probs2(device, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, output_choices: List[str], prob_dict: dict) -> List[float]:
    num_choices = len(output_choices)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_num_tokens = prompt_ids.size()[-1]
    combined_output_lst = [prompt + output for output in output_choices]
    combined_pt_lst = [tokenizer(c, return_tensors="pt").input_ids.to(device) for c in combined_output_lst]
    choices_probs = []
    for i in range(len(combined_pt_lst)):
        output_choice_txt = output_choices[i]
        input = combined_pt_lst[i]
        if output_choice_txt in prob_dict.keys():
            choices_probs.append(prob_dict[output_choice_txt])
        else:
            with torch.no_grad():
                logits = model(input).logits.squeeze()[prompt_num_tokens-1:-1]
            expected_out_ids = input.squeeze()[prompt_num_tokens:].unsqueeze(dim=1)
            probs = F.softmax(logits, dim=1)
            pred_probs = torch.gather(probs, dim=1, index=expected_out_ids).squeeze()
            choices_probs.append(torch.prod(pred_probs).item())
    return choices_probs


def get_completion_openai(model_name: str, prompt, max_tokens, logprobs=10):
    completion = openai_client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=max_tokens,
                    logprobs=logprobs
                )
    return completion

def not_prefix(output_choices: List[str], prev_lst: List[str]) -> bool:
    for context, _ in prev_lst:
        for expected_out in output_choices:
            if expected_out.startswith(context):
                return False
    return True

def get_probs_openai(model_name: str, prompt: str, output_choices: List[str], num_beams=10, max_tokens=None, lower_bound1=0.001, lower_bound2=0.001) -> list:
    if max_tokens == None:
        tokenizer = tiktoken.encoding_for_model(model_name)
        out_num_tokens = [len(tokenizer.encode(output)) for output in output_choices]
        max_tokens = int(max(out_num_tokens) * 1.5)
    prev_lst = [("", 1.0)]
    final_dict = {}
    for i in range(max_tokens):
        cur_lst = []
        for context, prev_prob in prev_lst:
            completion = get_completion_openai(model_name, prompt+context, max_tokens=1, logprobs=num_beams)
            try:
                top_logprob_dict = completion.choices[0].logprobs.top_logprobs[0]
            except:
                continue
            top_logprob_lst = [(k, v) for k, v in top_logprob_dict.items()]
            for token, logprob in top_logprob_lst:
                prob = np.exp(logprob)
                if context+token in output_choices:
                    if context+token in final_dict.keys():
                        final_dict[context+token] = final_dict[context+token] + prev_prob * prob
                    else:
                        final_dict[context+token] = prev_prob * prob
                else:
                    if prob > lower_bound1 and prev_prob * prob > lower_bound2:
                        cur_lst.append((context+token, prev_prob * prob))
        if len(cur_lst) > 0:
            prev_lst = sorted(cur_lst, key=lambda x: x[1], reverse=True)[:num_beams]
            if not_prefix(output_choices, prev_lst):
                break
        else:
            break
    choices_probs = []
    for output in output_choices:
        if output not in final_dict.keys():
            choices_probs.append(0.0)
        else:
            choices_probs.append(final_dict[output])
    return choices_probs

def get_top_probs2(device, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, num_beams, max_tokens: int, terminator="\n", eps=0.01)-> list:
    tmp_lst = tokenizer(terminator, add_special_tokens=False).input_ids
    assert len(tmp_lst) == 1
    terminator_id = tmp_lst[0]
    
    prev_lst = [("", 1.0)]
    final_dict = {}
    for i in range(max_tokens):
        cur_lst = []
        for context, prev_prob in prev_lst:
            pac = prompt + context
            pac_ids = tokenizer(pac, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = model(pac_ids).logits.squeeze()[-1]
            probs = F.softmax(logits, dim=0)
            sorted_probs, indices = torch.sort(probs, descending=True)
            bool1 = True
            if indices[0] == terminator_id:
                if probs[0] > 0.5: # if largest probable next is to terminate and the prob > 0.5 then no need to go through other path
                    bool1 = False
                if context in final_dict.keys():
                    final_dict[context] = final_dict[context] + prev_prob
                else:
                    final_dict[context] = prev_prob
            if bool1:
                for j in range(num_beams):
                    prob = sorted_probs[j].item()
                    token_id = indices[j]
                    token = tokenizer.decode(token_id)
                    cur_lst.append((context+token, prev_prob * prob))
        if len(cur_lst) > 0:
            prev_lst = sorted(cur_lst, key=lambda x: x[1], reverse=True)[:num_beams]
            if prev_lst[0][1] < eps:
                break
        else:
            break
    return final_dict, sorted(final_dict.items(), key=lambda x: x[1], reverse=True)

def prompt_model(model_alias, model_kwargs, prompt_config: PromptConfig, result_dir : str, pretrained_dir=None):
    model_res_dir = f"{result_dir}/{model_alias}"
    os.makedirs(model_res_dir, exist_ok=True)
    set_seed(prompt_config.seed)

    model_name = model_kwargs['name']
    if 'api' in model_kwargs:
        api = model_kwargs['api']
    else:
        api = None
    if 'quantization_config' in model_kwargs:
        quantization_kwargs = model_kwargs['quantization_config']
        quantization_config = BitsAndBytesConfig(**quantization_kwargs)
    else:
        quantization_config = None

    if "32B" in model_name or "70B" in model_name or "72B" in model_name:
        device_map = 'auto'
    else:
        device_map = {"": 0}

    torch.cuda.empty_cache()
    
    if api:
        model = None
        tokenizer = None
    else:
        if pretrained_dir:
            model = AutoModelForCausalLM.from_pretrained(
                f"{PRETRAINED_MODELS_DIR}/{model_name}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                use_cache=True,
                device_map = device_map,
                quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(f"{PRETRAINED_MODELS_DIR}/{model_name}")
        else: 
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                use_cache=True,
                device_map = device_map,
                quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []
    is_topk_lst = []
    kl_lst = []
    target_tensor = torch.Tensor(prompt_config.actual_distribution)
    for i in tqdm(range(prompt_config.num_exper)):
        prompt, output_choices = generate_prompt(prompt_config, tokenizer)
        if i == 0:
            with open(f"{model_res_dir}/first_prompt.txt", "w", encoding="utf-8") as outfile:
                outfile.write(prompt+"\n\n"+str(output_choices))
        if api:
            if api == "openai":
                result = get_probs_openai(model_name, prompt, output_choices)
            else:
                raise NotImplementedError
        else:
            if prompt_config.calculate_topk:
                if prompt_config.max_tokens == 0:
                    raise RuntimeError("calculate_topk is true but max_tokens==0")
                else:
                    prob_dict, top_probs = get_top_probs2(prompt_config.device, model, tokenizer, prompt, prompt_config.num_beams, prompt_config.max_tokens)
                    result = get_probs2(prompt_config.device, model, tokenizer, prompt, output_choices, prob_dict)
            else:
                result = get_probs(prompt_config.device, model, tokenizer, prompt, output_choices)
        result_tensor = torch.Tensor(result)
        kl = F.kl_div(result_tensor.log(), target_tensor, None, None, 'sum')
        if api or (not prompt_config.calculate_topk):
            k_prob = None
            choices_is_topk = None
        else:
            k_prob = top_probs[prompt_config.num_tasks-1][1]
            choices_is_topk = [int(result[i] >= k_prob - 1e-5) for i in range(len(result))]
        results.append(result)
        is_topk_lst.append(choices_is_topk if not api else 0)
        kl_lst.append(kl)
    results_tensor = torch.Tensor(results)
    if api or (not prompt_config.calculate_topk):
        is_topk_tensor = torch.Tensor([0])
    else:
        is_topk_tensor = torch.Tensor(is_topk_lst)
    kl_tensor = torch.Tensor(kl_lst)
    torch.save(results_tensor, f"{model_res_dir}/results.pt")
    torch.save(is_topk_tensor, f"{model_res_dir}/is_topk.pt")
    mean_tensor = torch.mean(results_tensor, dim=0)
    var_tensor = torch.var(results_tensor, dim=0)
    sorted_tensor_mean = torch.mean(torch.sort(results_tensor, dim=1, descending=True).values, dim=0)
    with open(f"{model_res_dir}/statistics.txt", "w", encoding="utf-8") as outfile:
        outfile.write(f"mean: {str(mean_tensor.tolist())}; other: {1-sum(mean_tensor.tolist())}\n")
        outfile.write("var: "+str(var_tensor.tolist())+"\n")
        outfile.write(f"sorted_tensor_mean: {str(sorted_tensor_mean.tolist())}; other: {1-sum(sorted_tensor_mean.tolist())}\n")
        if prompt_config.calculate_topk and prompt_config.max_tokens > 0 and not api:
            outfile.write(f"is_topk_mean: {str(torch.mean(is_topk_tensor, dim=0).tolist())}; {sum(torch.mean(is_topk_tensor, dim=0).tolist())}\n")
        outfile.write(f"kl_div_mean: {str(kl_tensor.mean())}")
    with open(f"{model_res_dir}/last_prompt.txt", "w", encoding="utf-8") as outfile:
        outfile.write(prompt)

def main(raw_config_path):
    with open(raw_config_path, 'r', encoding="utf-8") as fin:
        raw_config = yaml.load(fin, Loader)
    assert raw_config is not None, "Config file is empty: " + raw_config_path

    cur_time = get_time()
    if 'dir_name' in raw_config.keys():
        assert '/' not in raw_config['dir_name']
        result_dir = f"results/{cur_time}-{raw_config['dir_name']}"
    else:
        result_dir = f"results/{cur_time}"
    os.mkdir(result_dir)
    with open(f"{result_dir}/config.yaml", "w") as fout:
        yaml.dump(raw_config, fout)
    
    prompt_config = parse_raw_config_for_prompt(raw_config)
    with open(f"{result_dir}/actual_distribution.txt", "w") as fout:
        fout.write(str(prompt_config.actual_distribution))

    model_aliases = raw_config['model_aliases']
    model_kwargs = raw_config['model_kwargs']
    pretrained_dir = raw_config['pretrained_dir']
    for model_alias in model_aliases:
        prompt_model(model_alias, model_kwargs[model_alias], prompt_config, result_dir, pretrained_dir=pretrained_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_config", help="raw_config file path", type=str, required=True)

    args = parser.parse_args()

    main(args.raw_config)