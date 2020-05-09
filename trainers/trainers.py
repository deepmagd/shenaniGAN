from trainers.stage_1_trainer import Stage1Trainer
from trainers.stage_2_trainer import Stage2Trainer


def get_trainer(stage):
    """ Get the trainer object which prepares and trains the adequate model """
    trainer = f'Stage{stage}Trainer'
    print(trainer)
    return eval(trainer)
