"""Mostly copied from https://github.com/sunnweiwei/RankGPT/blob/main/rank_gpt.py"""

import copy
import json
import os
import re

import time
from argparse import ArgumentParser
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError
from tqdm import tqdm

load_dotenv()

np.random.seed(42)


class SafeOpenai:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def catch_timeout(func):
        def wrapper(*args, **kwargs):
            tries = 0
            while True:
                try:
                    tries += 1
                    return func(*args, **kwargs)
                except APITimeoutError as e:
                    if tries > 5:
                        raise e

        return wrapper

    @staticmethod
    def catch_rate_limit(func):
        def wrapper(*args, **kwargs):
            tries = 0
            while True:
                try:
                    tries += 1
                    return func(*args, **kwargs)
                except Exception as e:
                    pattern = r"Please try again in (\d+m)?(\d+.?\d*)(\w+)\."
                    match = re.search(pattern, str(e))
                    if match is None or tries > 5:
                        raise e
                    minutes = 0
                    if match.group(1):
                        minutes = float(match.group(1).replace("m", ""))
                    duration = float(match.group(2))
                    unit = match.group(3)
                    if unit == "s":
                        seconds = duration
                    elif unit == "ms":
                        seconds = duration / 1000
                    else:
                        raise ValueError(f"Unknown unit {unit}")
                    seconds = minutes * 60 + seconds
                    for _ in tqdm(
                        range(int(seconds) + 1),
                        position=1,
                        leave=False,
                        desc="Waiting for rate limit",
                    ):
                        time.sleep(1)

        return wrapper

    @catch_timeout
    @catch_rate_limit
    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        completion = self.client.chat.completions.create(*args, **kwargs, timeout=60)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    @catch_timeout
    @catch_rate_limit
    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        completion = self.client.completions.create(*args, **kwargs)
        if return_text:
            completion = completion["choices"][0]["text"]
        return completion


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-4-1106-preview":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message, tokens_per_name = 0, 0

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def max_tokens(model):
    if model == "gpt-4-1106-preview":
        return 128_000
    if "gpt-4" in model:
        return 8192
    else:
        return 4096


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({"query": topics, "hits": []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if "title" in content:
                content = (
                    "Title: " + content["title"] + " " + "Content: " + content["text"]
                )
            else:
                content = content["contents"]
            content = " ".join(content.split())
            ranks[-1]["hits"].append(
                {
                    "content": content,
                    "qid": qid,
                    "docid": hit.docid,
                    "rank": rank,
                    "score": hit.score,
                }
            )
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]["title"]
            ranks.append({"query": query, "hits": []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if "title" in content:
                    content = (
                        "Title: "
                        + content["title"]
                        + " "
                        + "Content: "
                        + content["text"]
                    )
                else:
                    content = content["contents"]
                content = " ".join(content.split())
                ranks[-1]["hits"].append(
                    {
                        "content": content,
                        "qid": qid,
                        "docid": hit.docid,
                        "rank": rank,
                        "score": hit.score,
                    }
                )
    return ranks


def get_prefix_prompt(query, num):
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
        },
        {
            "role": "user",
            "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(
    item, rank_start=0, rank_end=100, model_name="gpt-3.5-turbo"
):
    query = item["query"]
    num = len(item["hits"][rank_start:rank_end])

    max_length = 300
    while True:
        messages = get_prefix_prompt(query, num)
        rank = 0
        for hit in item["hits"][rank_start:rank_end]:
            rank += 1
            content = hit["content"]
            content = content.replace("Title: Content: ", "")
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = " ".join(content.split()[: int(max_length)])
            messages.append({"role": "user", "content": f"[{rank}] {content}"})
            messages.append(
                {"role": "assistant", "content": f"Received passage [{rank}]."}
            )
        messages.append({"role": "user", "content": get_post_prompt(query, num)})

        if (
            num_tokens_from_messages(messages, model_name)
            <= max_tokens(model_name) - 200
        ):
            break
        else:
            max_length -= 1
    return messages


def run_llm(messages, api_key: str, model_name="gpt-3.5-turbo"):
    agent = SafeOpenai(api_key)
    response = agent.chat(
        model=model_name, messages=messages, temperature=0, return_text=True
    )
    return response


def clean_response(response: str):
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item["hits"][rank_start:rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item["hits"][j + rank_start] = copy.deepcopy(cut_range[x])
        if "rank" in item["hits"][j + rank_start]:
            item["hits"][j + rank_start]["rank"] = cut_range[j]["rank"]
        if "score" in item["hits"][j + rank_start]:
            item["hits"][j + rank_start]["score"] = cut_range[j]["score"]
    return item


def permutation_pipeline(
    item,
    api_key: str,
    rank_start: int = 0,
    rank_end: int = 100,
    model_name="gpt-3.5-turbo",
):
    messages = create_permutation_instruction(
        item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name
    )  # chan
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(
        item, permutation, rank_start=rank_start, rank_end=rank_end
    )
    return item


def sliding_windows(
    item,
    api_key: str,
    rank_start: int = 0,
    rank_end: int = 100,
    window_size: int = 20,
    step: int = 10,
    model_name="gpt-3.5-turbo",
):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(
            item,
            api_key=api_key,
            rank_start=start_pos,
            rank_end=end_pos,
            model_name=model_name,
        )
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def write_eval_file(rank_results, file):
    with open(file, "w") as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]["hits"]
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def to_df(item):
    data = []
    for rank, hit in enumerate(item["hits"]):
        rank += 1
        depth = len(item["hits"])
        data.append(
            {
                "query_id": item["query_id"],
                "Q0": "Q0",
                "doc_id": hit["doc_id"],
                "rank": rank,
                "score": depth - rank + 1,
                "system": "rank_gpt",
            }
        )
    return pd.DataFrame(data)


def main():
    parser = ArgumentParser()

    parser.add_argument("--run_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--ir_dataset", type=str)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--add_positive_passages", action="store_true")

    args = parser.parse_args()

    dataset = ir_datasets.load(args.ir_dataset)
    run = pd.read_csv(
        args.run_file,
        sep=r"\s+",
        names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        dtype={"query_id": str, "doc_id": str, "rank": float},
    )
    run = run.drop(["Q0", "system", "score"], axis=1)
    run = run.groupby("query_id").head(args.k).reset_index(drop=True)

    if args.output_file.exists():
        reranked_run = pd.read_csv(
            args.output_file,
            sep=r"\s+",
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
            dtype={"query_id": str, "doc_id": str, "rank": float},
        )
        reranked_query_ids = reranked_run["query_id"].unique()
        print(
            f"Already found reranked run. "
            f"Removing {reranked_query_ids.shape[0]} "
            "already reranked queries from input run."
        )
        run = run[~run["query_id"].isin(reranked_query_ids)]
    else:
        if not args.output_file.parent.exists():
            raise ValueError("Output dir does not exist.")
        reranked_run = pd.DataFrame()

    docs_store = dataset.docs_store()
    queries = (
        pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
    )
    if args.add_positive_passages:
        qrels = pd.DataFrame(dataset.qrels_iter())
        qrels = qrels.drop("iteration", axis=1, errors="ignore")
        qrels = qrels[qrels["relevance"] > 0]
        qrels = qrels.loc[qrels["query_id"].isin(run["query_id"].unique())]
        run = run.merge(qrels, on=["query_id", "doc_id"], how="outer")
        random_ranks = np.random.rand(run["rank"].isna().sum()) * args.k
        run.loc[run["rank"].isna(), "rank"] = random_ranks
        run["rank"] = run.groupby("query_id")["rank"].rank(method="dense").astype(int)
        run = run.sort_values(["query_id", "rank"])
        run = run.groupby("query_id").head(args.k).reset_index(drop=True)

    rank_results = []
    for query_id, ranking in run.groupby("query_id"):
        results = {"query": queries[query_id], "query_id": query_id}
        hits = []
        for doc_id in ranking["doc_id"]:
            doc = docs_store.get(doc_id)
            hits.append({"content": doc.default_text(), "doc_id": doc_id})
        results["hits"] = hits
        rank_results.append(results)

    key = os.environ.get("OPENAI_API_KEY", args.api_key)
    if key is None:
        raise ValueError("Please provide an OpenAI API key.")

    for item in tqdm(rank_results):
        new_item = sliding_windows(
            item,
            rank_start=0,
            rank_end=args.k,
            window_size=args.window_size,
            step=args.step,
            model_name=args.model_name,
            api_key=args.api_key,
        )
        new_df = to_df(new_item)
        reranked_run = pd.concat((reranked_run, new_df))
        new_df.to_csv(args.output_file, sep="\t", index=False, header=False, mode="a")


if __name__ == "__main__":
    main()
