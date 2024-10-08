from huggingface_hub import snapshot_download

PRETRAINED_MODELS_DIR = "<path-to-pretrained-models>"
HF_TOKEN = "<your-hf-token>"

model_lst = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-72B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
]

for model_repo in model_lst:
    model_name = model_repo.split("/")[-1]
    snapshot_download(repo_id=model_repo,
    local_dir=f"{PRETRAINED_MODELS_DIR}/{model_name}",
    token=HF_TOKEN
)