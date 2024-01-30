from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal

import lightning.pytorch as pl
import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

from set_encoder.model.set_encoder import SetEncoderClassFactory
from set_encoder.model import loss_utils


class SetEncoderModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        depth: int = 100,
        other_doc_attention: bool = False,
        rank_position_embeddings: bool | Literal["random", "sorted"] = False,
        freeze_position_embeddings: bool = False,
        loss_function: loss_utils.LossFunc = loss_utils.RankNet(),
        compile_model: bool = True,
        use_flash: bool = True,
    ) -> None:
        """SetEncoderModule which wraps a pretrained BERT-based model and adds
        overwrites the forward functions to enable cross-document information.

        Args:
            model_name_or_path (str): name of the pretrained model or path to the model.
            depth (int, optional): maximum re-ranking depth. Defaults to 100.
            other_doc_attention (bool, optional): toggle to enable cross-document
                information (sharing of CLS tokens). Defaults to False.
            rank_position_embeddings (bool | Literal['random', 'sorted'], optional):
                toggle to turn on rank position embeddings, making the model aware of
                the initial positions in the initial ranking. Defaults to False.
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
            rank_position_embeddings=rank_position_embeddings,
            use_flash=use_flash,
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
        loss = self.loss_function.compute(logits, labels)
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
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        num_docs = batch["num_docs"]
        out = self.forward(input_ids, attention_mask, num_docs)
        logits = out.logits

        dataset_name = ""
        first_stage = ""
        try:
            ir_dataset_path = Path(
                self.trainer.datamodule.val_ir_dataset_paths[dataloader_idx]
            )
            dataset_name = ir_dataset_path.stem + "-"
            first_stage = ir_dataset_path.parent.name + "-"
        except RuntimeError:
            pass

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
        optimal_labels = torch.nn.utils.rnn.pad_sequence(
            torch.split(batch["optimal_labels"], num_docs),
            batch_first=True,
            padding_value=loss_utils.PAD_VALUE,
        )
        ranks = torch.argsort(torch.argsort(logits, dim=1, descending=True)) + 1

        val = loss_utils.get_ndcg(ranks, labels, k=10, optimal_labels=optimal_labels)
        metric_name = "ndcg@10"
        self.validation_step_outputs.append(
            (f"{first_stage}{dataset_name}{metric_name}", val)
        )
        val = loss_utils.get_mrr(ranks, labels)
        metric_name = "mrr@max"
        self.validation_step_outputs.append(
            (f"{first_stage}{dataset_name}{metric_name}", val)
        )

    def on_validation_epoch_end(self) -> None:
        aggregated = defaultdict(list)
        for key, value in self.validation_step_outputs:
            aggregated[key].extend(value)

        self.validation_step_outputs.clear()

        for key, value in aggregated.items():
            stacked = torch.stack(value)
            stacked[torch.isnan(stacked)] = 0
            self.log(key, stacked.mean(), sync_dist=True)

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
