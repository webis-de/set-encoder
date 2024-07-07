#!/usr/bin/env python3
import argparse
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import ir_datasets
tira = Client()
import gzip
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Load rerank file')

    parser.add_argument('--input-dataset', type=str, help='The directory with the input data (i.e., a queries.jsonl and a documents.jsonl file).', required=True)
    parser.add_argument('--input-run', type=str, help='The input run to re-rank.', required=True)
    parser.add_argument('--top-k', type=int, help="how many documents to rerank", required=True)
    parser.add_argument('--output', type=str, help='The output will be stored in this directory.', required=True)

    return parser.parse_args()

def rerank_df(input_dataset, input_run, top_k):
    plaidX = pd.read_csv(input_run, sep="\s+", names=["qid", "q0", "docno", "rank", "score", "system"], dtype={"qid": str, "docno": str})
    plaidX = plaidX[plaidX['rank'] <= top_k]
    dataset = ir_datasets.load(input_dataset)
    queries = {i.query_id: i.default_text() for i in dataset.queries_iter()}
    docs_store = dataset.docs_store()

    plaidX['query'] = plaidX['qid'].apply(lambda i: queries[i])
    plaidX['text'] = plaidX['docno'].apply(lambda i: docs_store.get(i).default_text())
    print(plaidX.head(10))
    return plaidX


if __name__ == '__main__':
    args = parse_args()
    df = rerank_df(args.input_dataset, args.input_run, args.top_k)

    with gzip.open(f'{args.output}/rerank.jsonl.gz', 'wt') as f:
        for _, i in df.iterrows():
            f.write(json.dumps({"qid": str(i['qid']), "query": i['query'], "original_query": {}, "docno": str(i['docno']), "text": i['text'], "original_document": {}, "rank": i['rank'], "score": i['score']}) + '\n')
