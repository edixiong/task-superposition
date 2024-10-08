# Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition

Implementation for paper "Everything Everywhere All at Once: LLMs can In-Context Learn Multiple Tasks in Superposition"

To install the environment, run
```
conda env create -f environment.yml
conda activate task_sup
```

To download pretrained models, fill in `PRETRAINED_MODELS_DIR` and `HF_TOKEN` in `download_models.py` and run
```
python download_models.py
```

To run the experiment as shown in Figure 2, fill in `OPENAI_API_KEY` and `PRETRAINED_MODELS_DIR` in run_experiments.py and run
```
mkdir results

python run_experiments.py --raw_config=config/add_translate.yaml
python run_experiments.py --raw_config=config/country.yaml
python run_experiments.py --raw_config=config/AB.yaml
python run_experiments.py --raw_config=config/word.yaml ## check the note below
```

**Important**

If you run inference on gpt-3.5-turbo-instruct, which is the default in all config files under folder `config`, there will be a charge on using this model. The current implementation to get probabilities of gpt-3.5-turbo-instruct uses the beam search. Check `get_probs_openai()` in `run_experiments.py` for more detail. If you do not want to run inference on gpt-3.5-turbo-instruct, you can change the config files in `config` folder.

**Note on experiment of Figure 2(d)**

To run the experiment
```
python run_experiments.py --raw_config=config/word.yaml
```
You need to install [random-word](https://github.com/vaibhavsingh97/random-word). To speed-up the process, please refer to [this issue](https://github.com/vaibhavsingh97/random-word/issues/89).

After the installation, uncomment
```
# from random_word import RandomWords
# r = RandomWords()
```
from `tasks.py` and run the experiment.