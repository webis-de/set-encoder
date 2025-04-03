# Set-Encoder

This repository contains the code for the paper: [`Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders`](https://webis.de/publications.html#schlatt_2025b) accepted at ECIR'25.

We use [`lightning-ir`](https://github.com/webis-de/lightning-ir) to train and fine-tune models. Download and install the library to use the code in this repository.

## Model Zoo

### General Purpose Re-Ranking

We provide the following pre-trained models for general purpose re-ranking.

To reproduce the results, run the following command:

```bash
lightning-ir re_rank --config ./configs/re-rank.yaml --model.model_name_or_path <MODEL_NAME> 
```

(nDCG@10 on TREC DL 19 and TREC DL 20)

| Model Name                                                                               | TREC DL 19 (BM25) | TREC DL 20 (BM25) | TREC DL 19 (ColBERTv2) | TREC DL 20 (ColBERTv2) |
| ---------------------------------------------------------------------------------------- | ----------------- | ----------------- | ---------------------- | ---------------------- |
| [webis/set-encoder-base](https://huggingface.co/webis/set-encoder-base)                  | 0.746             | 0.704             | 0.781                  | 0.768                  |
| [webis/set-encoder-large](https://huggingface.co/webis/set-encoder-large)                | 0.750             | 0.722             | 0.789                  | 0.791                  |


### Novelty-Aware Re-Ranking

We provide the following fine-tuned models for novelty-aware re-ranking.

To reproduce the results, run the following command:

```bash
lightning-ir re_rank --config ./configs/re-rank-novelty.yaml --model.model_name_or_path <MODEL_NAME> 
```

(alpha nDCG@10, alpha=0.99 on TREC DL 19 and TREC DL 20)

| Model Name                                                                               | TREC DL 19 (BM25) | TREC DL 20 (BM25) | TREC DL 19 (ColBERTv2) | TREC DL 20 (ColBERTv2) |
| ---------------------------------------------------------------------------------------- | ----------------- | ----------------- | ---------------------- | ---------------------- |
| [webis/set-encoder-novelty-base](https://huggingface.co/webis/set-encoder-novelty-base)  | 0.805             | 0.721             | 0.821                  | 0.803                  |

## Fine-Tuning

Pre-fine-tuning (first stage fine-tuning using positive samples from MS MARCO and hard-negatives sampled using ColBERTv2 with Duplicate-Aware InfoNCE) of a Set-Encoder model can be done using the following command.

```bash
lightning-ir fit --config ./configs/pre-finetune.yaml
```

The model can be further fine-tuned (second stage fine-tuning using the RankDistiLLM or RankDistiLLM-Novelty dataset with RankNet or Novelty-Aware RankNet loss) using the following command. The model checkpoint from the pre-fine-tuning stage can be used as a starting point.

```bash
lightning-ir fit --config ./configs/fine-tune.yaml
lightning-ir fit --config ./configs/fine-tune-novelty.yaml 
```

## Citation

If you use this code or the models in your research, please cite our paper:

```bibtex
@InProceedings{schlatt:2025,
  address =                  {Berlin Heidelberg New York},
  author =                   {Ferdinand Schlatt and Maik Fr{\"o}be and Harrisen Scells and Shengyao Zhuang and Bevan Koopman and Guido Zuccon and Benno Stein and Martin Potthast and Matthias Hagen},
  booktitle =                {Advances in Information Retrieval. 47th European Conference on IR Research (ECIR 2025)},
  doi =                      {10.1007/978-3-031-88711-6_1},
  month =                    apr,
  publisher =                {Springer},
  series =                   {Lecture Notes in Computer Science},
  site =                     {Lucca, Italy},
  title =                    {{Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking with Cross-Encoders}},
  year =                     2025
}
