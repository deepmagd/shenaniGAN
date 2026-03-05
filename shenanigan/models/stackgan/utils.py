from shenanigan.models.stackgan.stage1 import Stage1Trainer
from shenanigan.models.stackgan.stage2 import Stage2Trainer

from typing import Union

_TRAINERS = {1: Stage1Trainer, 2: Stage2Trainer}


def get_trainer(stage: int) -> Union[Stage1Trainer, Stage2Trainer]:
    """ Get the trainer object which prepares and trains the adequate model """
    if stage not in _TRAINERS:
        raise ValueError(f"Invalid stage: {stage}. Expected 1 or 2.")
    return _TRAINERS[stage]
