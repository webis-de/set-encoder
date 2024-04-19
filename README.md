# Set-Encoder

This repository contains the code for the paper: `Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders`.

## Note

The repository is undergoing a major refactoring. The last stable version can be found under commit [`ef99e78`](https://github.com/webis-de/set-encoder/tree/ef99e78e2b40cfce055aa55a27d7ca0c40cf53b4).

## Data

Training data must be generated prior to fine-tuning the model. The training code uses TREC-style run files to sample training data. We provide run files for ColBERTv2 and RankGPT-4-Turbo [here](https://zenodo.org/records/10952882). The run files are generated using the instructions below.

To generate ColBERTv2 run files required for first-stage fine-tuning, run the following command:

```sh
python set_encoder/data/create_baseline_runs.py \
    --ir_datasets msmarco-passage/train/judged \
    --run_dir data/baseline-runs \
    --index_dir data/indexes \
    --checkpoint_path colbert-ir/colbertv2.0
```

To generate the fine-tuning data for the second stage, run (we first randomly sampled a subset of 1000 queries from the ColBERTv2 run):

```sh
python rank_gpt.py \
    --run_file data/baseline-runs/colbert/__sampled__msmarco-pasasge-train-judged.run \
    --output_file data/baseline-runs/rankgpt-4-turbo \
    --ir_dataset msmarco-passage/train/judged \
    --api_key {YOUR_OPENAI_API_KEY} \
    --model_name gpt-4-1106-preview \
    --window_size 100
```

## Fine-tuning

The main entry point for fine-tuning and inference is `main.py`. To train a model using ColBERTv2 hard negatives, set the correct paths for the training run files in `set_encoder/configs/msmarco-passage-colbert.yaml` and run:

```sh
python main.py fit \
    --config set_encoder/configs/colbert-trainer.yaml \
    --config set_encoder/configs/set-encoder.yaml \
    --config set_encoder/configs/optimizer.yaml \
    --config set_encoder/configs/msmarco-passage-colbert.yaml
```

To continue with LLM-distillation fine-tuning, set the correct paths for the training run files in `set_encoder/configs/msmarco-passage-rankgpt-4-turbo.yaml` and update the model path in `set_encoder/configs/set_encoder.yaml` and run:

```sh
python main.py fit \
    --config set_encoder/configs/rankgpt-trainer.yaml \
    --config set_encoder/configs/set-encoder.yaml \
    --config set_encoder/configs/optimizer.yaml \
    --config set_encoder/configs/msmarco-passage-rankgpt-4-turbo.yaml
```