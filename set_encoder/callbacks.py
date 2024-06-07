from pathlib import Path
from lightning import Trainer
from lightning_ir import ReRankCallback


class ParentReRankCallback(ReRankCallback):

    def get_run_path(self, trainer: Trainer, dataset_idx: int) -> Path:
        run_path = super().get_run_path(trainer, dataset_idx)
        first_stage = Path(
            trainer.datamodule.inference_datasets[dataset_idx].run_path
        ).parent.name
        run_path = run_path.with_name(f"{first_stage}_{run_path.name}")
        return run_path
