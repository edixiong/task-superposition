from dataclasses import dataclass, field
from typing import List


@dataclass
class PromptConfig:
    num_exper : int
    num_examples : int
    num_tasks: int
    max_tokens: int
    num_beams: int
    calculate_topk: bool
    task_funcs_dict: dict
    question_name: str
    question_kwargs: dict
    task_examples: List[int]
    actual_distribution: List[float]
    order: str
    prompt_template: str
    device: str
    seed: int
    

def parse_raw_config_for_prompt(raw_config : dict):
    # parse raw_config
    num_exper = raw_config['num_exper']
    num_examples = raw_config['num_examples']
    num_tasks = raw_config['num_tasks']
    max_tokens = raw_config['max_tokens']
    num_beams = raw_config['num_beams']
    calculate_topk = raw_config['calculate_topk']
    task_funcs_dict = raw_config['task_funcs_dict']
    question_name = raw_config['question']['name']
    question_kwargs = raw_config['question']['kwargs']
    distribution = [eval(num) if type(num)==str else num for num in raw_config['distribution']]
    if len(distribution) != num_tasks:
        raise ValueError("length of distribution is not the same as the length of task_func_names")
    if 1 - sum(distribution) > 1e-5 or sum(distribution) - 1 > 1e-5:
        raise ValueError("distribution does not sum to 1")
    task_examples = [int(frac * num_examples) for frac in distribution]
    if sum(task_examples) > num_examples:
        raise ValueError("distribution sums to more than 1")
    elif sum(task_examples) < num_examples:
        num_diff = num_examples - sum(task_examples)
        i = 0
        while num_diff > 0 and i < len(task_examples):
            if not (num_examples * distribution[i]).is_integer():
                task_examples[i] += 1
                num_diff -= 1
            i += 1
        if sum(task_examples) < num_examples:
            raise ValueError("distribution sums to less than 1")
    order = raw_config['order']
    device = raw_config['device']
    seed = raw_config['seed']
    
    prompt_template_name = raw_config['prompt_template_name']
    with open(f"prompt/{prompt_template_name}.prompt", "r") as f:
        prompt_template = f.read()

    config = PromptConfig(num_exper=num_exper, 
                          num_examples=num_examples,
                          num_tasks=num_tasks,
                          max_tokens=max_tokens,
                          num_beams=num_beams,
                          calculate_topk=calculate_topk,
                          task_funcs_dict=task_funcs_dict,
                          question_name=question_name,
                          question_kwargs=question_kwargs,
                          task_examples=task_examples,
                          actual_distribution=[num_examples / sum(task_examples) for num_examples in task_examples],
                          order=order,
                          prompt_template=prompt_template,
                          device=device,
                          seed=seed)
    return config
