# flan-collection

## Requirements

```bash
conda create -y -n flan python=3.10
conda activate flan
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pre-commit install
```

Then, set `HF_HOME`, `HF_DATASETS_CACHE` and `TRANSFORMERS_CACHE`, e.g.,
```bash
export HF_HOME="<some dir>/.cache/huggingface/"
export HF_DATASETS_CACHE="<some dir>/.cache/huggingface/datasets/"
export TRANSFORMERS_CACHE="<some dir>/.cache/huggingface/transformers/"
```

## Download the Flan Collection
```bash
bash scripts/download_flan_collection.sh
```

## Finetuning T5

## Set up the environment
```bash
conda create flan python=3.11
conda activate flan
pip install -r t5-reqs/requirements.txt
```
## Start the finetuning 

```bash
sbatch scripts/run_t5_finetune.sh
```
Inside the run_t5_finetune.sh file be sure to change the export HF_HOME="<some dir>/.cache/huggingface/"
Specify the parameters for finetuning
You might have to pip install accelerate if you face any issues

## Run the evaluation script

You will have to download the data directory from this link : https://people.eecs.berkeley.edu/~hendrycks/data.tar
Untar the directory at the root level 
```bash
tar -xf data.tar
```
You can the evaluation script with a different model/model checkpoint by adding the flag to the python command in run_eval.sh
for e.g 
```bash
src/flan-eval.py --model "YOUR_MODEL"
```