from typing import Optional

from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment


class CustomWandbLogger(WandbLogger):
    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir
