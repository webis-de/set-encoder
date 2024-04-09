from typing import Any, Dict, Sequence

import pandas as pd
import pyndeval
from trectools import TrecEval, TrecQrel, TrecRun


ADHOC_METRICS = {"NDCG", "MRR", "UNJ", "rNDCG"}
DIVERSITY_METRICS = {"ERR-IA", "nERR-IA", "alpha-nDCG"}


def evaluate_run(
    run_df: pd.DataFrame, qrels_df: pd.DataFrame, metrics: Dict[str, Dict[str, Any]]
):
    run_df = run_df.rename(
        {"query_id": "query", "Q0": "q0", "doc_id": "docid", "run_name": "system"},
        axis=1,
    )
    adhoc_metrics = {
        metric: kwargs
        for metric, kwargs in metrics.items()
        if metric.split("@")[0] in ADHOC_METRICS
    }
    diversity_metrics = {
        metric: kwargs
        for metric, kwargs in metrics.items()
        if metric.split("@")[0] in DIVERSITY_METRICS
    }
    adhoc_values = evaluate_adhoc(adhoc_metrics, run_df, qrels_df)
    diversity_values = evaluate_diversity(diversity_metrics, run_df, qrels_df)
    values = pd.concat([adhoc_values, diversity_values], axis=1).fillna(0)
    return values


def evaluate_adhoc(
    full_metrics: Dict[str, Any], run_df: pd.DataFrame, qrels_df: pd.DataFrame
) -> pd.DataFrame:
    run = TrecRun()
    qrels = TrecQrel()
    run.run_data = run_df
    qrels_df = qrels_df.groupby(["query", "docid"])["rel"].max().reset_index()
    qrels.qrels_data = qrels_df
    trec_eval = TrecEval(run, qrels)
    metric_to_func = {
        "NDCG": "get_ndcg",
        "rNDCG": "get_ndcg",
        "MRR": "get_reciprocal_rank",
        "UNJ": "get_unjudged",
    }
    dfs = []
    for full_metric, kwargs in full_metrics.items():
        metric, depth = full_metric.split("@")
        depth = depth.split("_")[0]
        depth = int(depth)
        func_name = metric_to_func[metric]
        func = getattr(trec_eval, func_name)
        df = func(depth, per_query=True, **kwargs)
        df = df.rename(lambda x: full_metric, axis=1)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df


def evaluate_diversity(
    full_metrics: Dict[str, Any], run_df: pd.DataFrame, qrels_df: pd.DataFrame
) -> pd.DataFrame:
    if "subtopic_id" not in qrels_df:
        return
    qrels = [
        pyndeval.SubtopicQrel(row.query, row.subtopic_id, row.docid, row.rel)
        for _, row in qrels_df.iterrows()
    ]
    run = [
        pyndeval.ScoredDoc(row.query, row.docid, row.score)
        for _, row in run_df.iterrows()
    ]
    metrics = [metric for metric in full_metrics if not metric.endswith("_UNJ")]
    values = pd.DataFrame(pyndeval.ndeval(qrels, run, measures=metrics)).T
    unj_metrics = [
        metric.rstrip("_UNJ") for metric in full_metrics if metric.endswith("_UNJ")
    ]
    if unj_metrics:
        run_df = (
            run_df.merge(
                qrels_df.loc[:, ["query", "docid", "rel"]].drop_duplicates(
                    subset=["query", "docid"]
                ),
                on=["query", "docid"],
                how="left",
            )
            .dropna(subset="rel")
            .drop(columns="rel")
        )
        run = [
            pyndeval.ScoredDoc(row.query, row.docid, row.score)
            for _, row in run_df.iterrows()
        ]
        unj_values = pd.DataFrame(pyndeval.ndeval(qrels, run, measures=unj_metrics)).T
        values = pd.concat([values, unj_values], axis=1)
    values.index.name = "query"
    return values
