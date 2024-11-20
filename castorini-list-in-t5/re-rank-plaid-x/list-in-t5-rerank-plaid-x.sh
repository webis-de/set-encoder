#!/usr/bin/env bash

INPUT_RUN=$(tira-cli download --dataset $1 --approach reneuir-2024/reneuir-baselines/plaid-x-retrieval)

/prepare-rerank-file-from-plaid-x.py --output /tmp/ --input-dataset $1 --input-run ${INPUT_RUN}/run.txt --top-k 100 

zcat /tmp/rerank.jsonl.gz |head -10

/run-notebook.py --input /tmp/ --output $2 --notebook /workspace/notebooks/run-list-in-t5.ipynb

