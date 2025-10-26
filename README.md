# PIXEL: ADAPTIVE STEERING VIA POSITION-WISE INJECTION WITH EXACT ESTIMATED LEVELS UNDER SUBSPACE CALIBRATION


# Setup

Install dependencies from requirements.txt (Python ≥ 3.10):

```bash
pip install -U pip
pip install -r requirements.txt
```
# Reproduce Our Results (TruthfulQA MC1/MC2)

Run one of the following to reproduce the results for each model:

 LLaMA-3-8B-Instruct

 ```bash
python experiments/truthfulqa_mc1mc2.py --preset llama
 ```
 Mistral-7B-v0.3

```bash
python experiments/truthfulqa_mc1mc2.py --preset mistral
 ```
 Qwen2-7B-Instruct

```bash
python experiments/truthfulqa_mc1mc2.py --preset qwen
 ```
