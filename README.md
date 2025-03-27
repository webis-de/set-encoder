# Set-Encoder

This repository contains the code for the paper: [`Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders`](https://webis.de/publications.html#schlatt_2025b) accepted at ECIR'25.

We use [`lightning-ir`](https://github.com/webis-de/lightning-ir) to train and fine-tune models. Download and install the library to use the code in this repository.

## Model Zoo

We provide the following pre-trained models:

| Model Name                                                          | TREC DL 19 (BM25) | TREC DL 20 (BM25) | TREC DL 19 (ColBERTv2) | TREC DL 20 (ColBERTv2) |
| ------------------------------------------------------------------- | ----------------- | ----------------- | ---------------------- | ---------------------- |
| [set-encoder-base](https://huggingface.co/webis/set-encoder-base)   | 0.724             | 0.710             | 0.788                  | 0.777                  |
| [set-encoder-large](https://huggingface.co/webis/set-encoder-large) | 0.727             | 0.735             | 0.789                  | 0.790                  |

## Inference

We recommend using the `lightning-ir` cli to run inference. The following command can be used to run inference using the `set-encoder-base` model on the TREC DL 19 and TREC DL 20 datasets:

```bash
lightning-ir re_rank --config configs/re-rank.yaml --config configs/set-encoder-finetuned.yaml --config configs/trec-dl.yaml
```

## Fine-Tuning

WIP
