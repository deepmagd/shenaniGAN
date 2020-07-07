from shenanigan.models.stackgan.stage1 import Stage1Trainer
from shenanigan.models.stackgan.stage2 import Stage2Trainer

from typing import Union


def get_trainer(stage: int) -> Union[Stage1Trainer, Stage2Trainer]:
    """ Get the trainer object which prepares and trains the adequate model """
    trainer = f'Stage{stage}Trainer'
    print(trainer)
    return eval(trainer)
