from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import ir_datasets
import lightning.pytorch as pl
import pandas as pd
import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

from set_encoder.data.ir_dataset_utils import DASHED_DATASET_MAP
from set_encoder.model import loss_utils
from set_encoder.model.validation_utils import evaluate_run
from set_encoder.model.set_encoder import SetEncoderClassFactory


class SetEncoderModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        depth: int = 100,
        other_doc_attention: bool = False,
        freeze_position_embeddings: bool = False,
        loss_function: loss_utils.LossFunc = loss_utils.RankNet(),
        compile_model: bool = True,
        use_flash: bool = True,
        fill_random_docs: bool = True,
    ) -> None:
        """SetEncoderModule which wraps a pretrained BERT-based model and adds
        overwrites the forward functions to enable cross-document information.

        Args:
            model_name_or_path (str): name of the pretrained model or path to the model.
            depth (int, optional): maximum re-ranking depth. Defaults to 100.
            other_doc_attention (bool, optional): toggle to enable cross-document
                information (sharing of CLS tokens). Defaults to False.
            freeze_position_embeddings (bool, optional): toggle to freeze positional
                encodings. Defaults to False.
            loss_function (loss_utils.LossFunc, optional): the loss function to apply.
                see loss_utils for options. Defaults to loss_utils.RankNet().
            compile_model (bool, optional): toggle to compile the model to improve
                computation time. Defaults to True.
        """
        super().__init__()
        self.loss_function = loss_function
        config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        model_class = transformers.AutoModelForSequenceClassification._model_mapping[
            type(config)
        ]
        SetEncoder = SetEncoderClassFactory(model_class)
        self.set_encoder = SetEncoder.from_pretrained(
            model_name_or_path,
            depth=depth,
            other_doc_attention=other_doc_attention,
            use_flash=use_flash,
            fill_random_docs=fill_random_docs,
        )
        if freeze_position_embeddings:
            for name, param in self.set_encoder.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False
                    break

        if compile_model:
            torch.compile(self.set_encoder)

        self.validation_step_outputs = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_docs: List[int],
    ) -> SequenceClassifierOutput:
        return self.set_encoder(
            input_ids, attention_mask=attention_mask, num_docs=num_docs
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]
        out = self.forward(input_ids, attention_mask, num_docs)

        logits = torch.nn.utils.rnn.pad_sequence(
            torch.split(out.logits.squeeze(1), num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            torch.split(batch["labels"], num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        subtopics = None
        if "subtopics" in batch:
            subtopics = torch.nn.utils.rnn.pad_sequence(
                torch.split(batch["subtopics"], num_docs),
                batch_first=True,
                padding_value=loss_utils.PAD_VALUE,
            )
        loss = self.loss_function.compute(logits, labels, subtopics)
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> List[torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]
        out = self.forward(input_ids, attention_mask, num_docs)
        logits = torch.split(out.logits.squeeze(1), num_docs)
        return logits

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        dataset_name = ""
        first_stage = ""
        try:
            ir_dataset_path = Path(
                self.trainer.datamodule.val_ir_dataset_paths[dataloader_idx]
            )
            dataset_name = ir_dataset_path.name[
                : -len("".join(ir_dataset_path.suffixes))
            ]
            first_stage = ir_dataset_path.parent.name
        except RuntimeError:
            return

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]

        out = self.forward(input_ids, attention_mask, num_docs)
        logits = out.logits.view(-1).tolist()
        query_ids = [
            query_id
            for query_idx, query_id in enumerate(batch["query_id"])
            for _ in range(num_docs[query_idx])
        ]
        doc_ids = [doc_id for doc_ids in batch["doc_ids"] for doc_id in doc_ids]

        self.validation_step_outputs.append(
            (
                f"{first_stage}/{dataset_name}",
                {"score": logits, "query_id": query_ids, "doc_id": doc_ids},
            )
        )

    def on_validation_epoch_end(self) -> None:
        aggregated = defaultdict(lambda: defaultdict(list))
        for dataset, value_dict in self.validation_step_outputs:
            for key, value in value_dict.items():
                aggregated[dataset][key].extend(value)

        self.validation_step_outputs.clear()

        for dataset, values in aggregated.items():
            run = pd.DataFrame(values)
            run["rank"] = run.groupby("query_id")["score"].rank(
                ascending=False, method="first"
            )
            run["Q0"] = "0"
            run["run_name"] = "set-encoder"
            dataset_id = dataset.split("/")[1]
            qrels = pd.DataFrame(
                ir_datasets.load(DASHED_DATASET_MAP[dataset_id]).qrels_iter()
            )
            qrels = qrels.rename(
                {"doc_id": "docid", "relevance": "rel", "query_id": "query"}, axis=1
            )
            metrics = {
                "NDCG@10": {},
                "NDCG@10_UNJ" : {"removeUnjudged": True},
                "UNJ@10": {},
                "alpha-nDCG@10": {},
                "alpha-nDCG@10_UNJ": {},
                "ERR-IA@10": {},
                "ERR-IA@10_UNJ": {},
            }

            values = evaluate_run(run, qrels, metrics)
            for metric, value in values.mean().items():
                self.log(f"{dataset}/{metric}", value)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            step = self.trainer.global_step
            self.set_encoder.config.save_step = step
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.set_encoder.save_pretrained(save_path)
            try:
                self.trainer.datamodule.tokenizer.save_pretrained(save_path)
            except:
                pass
