# SIGIR 24 Set-Encoder

This repository contains the code for the paper SIGIR'24 submission: `Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders`.

Prior to fine-tuning a model (we will release fine-tuned models on the HuggingFace Hub in case of acceptance), the training data needs to be created/pre-processed (all data will be released on Zenodo in case of acceptance, but is too large to share in this repository). To generate ColBERTv2 run files required for first-stage fine-tuning, run the following command:

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

The main entry point for fine-tuning and inference is `main.py`. To train a model modify the configuration files under `set_encoder/configs/cli` and run:

```sh
python main.py fit \
    --config set_encoder/configs/cli/trainer.yaml \
    --config set_encoder/configs/cli/set-encoder.yaml \
    --config set_encoder/configs/cli/optimizer.yaml \
    --config set_encoder/configs/cli/msmarco-passage.yaml
```

