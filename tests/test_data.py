from set_encoder.data.datamodule import SetEncoderDataModule, ListwiseDataset


def test_eval_dataset(dataset: ListwiseDataset):
    dataset.relevant_sampling_strategy = "first"
    dataset.non_relevant_sampling_strategy = "first"
    dataset.num_relevant_samples = -1
    sample = next(iter(dataset))
    dataset_id = next(iter(dataset.runs.keys()))
    df = dataset.runs[dataset_id].copy()
    df = df.loc[df["query_id"] == sample["query_id"]]
    df = df.loc[df["rank"] > 0]
    qrels = dataset.qrels[dataset_id].get_group(sample["query_id"])
    doc_ids = df["doc_id"].head(len(sample["doc_ids"])).tolist()
    labels = qrels.set_index("doc_id")["relevance"].to_dict()
    for idx, doc_id in enumerate(sample["doc_ids"]):
        assert doc_id in doc_ids
        assert labels[doc_id] == sample["labels"][idx]


def test_random_dataset(dataset: ListwiseDataset):
    dataset.non_relevant_sampling_strategy = "random"
    for i in range(10):
        iterator = iter(dataset)
        while True:
            try:
                sample = next(iterator)
            except StopIteration:
                break
    sample = next(iter(dataset))
    assert len(sample["labels"]) == dataset.sample_size


def test_first_neg_dataset(dataset: ListwiseDataset):
    dataset.non_relevant_sampling_strategy = "random_first_neg"
    sample = next(iter(dataset))
    assert len(sample["labels"]) == dataset.sample_size


def test_ignore_neg_dataset(dataset: ListwiseDataset):
    dataset.non_relevant_sampling_strategy = "random_ignore_neg"
    sample = next(iter(dataset))
    assert len(sample["labels"]) == dataset.sample_size


def test_first_and_ignore_neg_dataset(dataset: ListwiseDataset):
    dataset.non_relevant_sampling_strategy = "random_first_neg_ignore_neg"
    sample = next(iter(dataset))
    assert len(sample["labels"]) == dataset.sample_size


def test_first_dataset(dataset: ListwiseDataset):
    dataset.non_relevant_sampling_strategy = "first"
    sample = next(iter(dataset))
    assert len(sample["labels"]) == dataset.sample_size


def test_listwise_datamodule(datamodule: SetEncoderDataModule):
    dataloader = datamodule.train_dataloader()
    sample = next(iter(dataloader))
    assert "query_input_ids" in sample
    assert "doc_input_ids" in sample
    assert "labels" in sample
