from trainers.stage_1_trainer import Stage1Trainer


TRAINERS_DICT = {
    "birds-with-text": "Stage1Trainer",
    "flowers-with-text": "Stage1Trainer",
    "xrays": "Stage1Trainer"
}
TRAINERS = list(TRAINERS_DICT.keys())

def get_trainer(dataset_name):
    """ Get the trainer object which prepares and trains
        the adequate model
    """
    if dataset_name in TRAINERS:
        trainer = TRAINERS_DICT[dataset_name]
        print(trainer)
        return eval(trainer)
    else:
        raise Exception('Invalid dataset name: {}.'.format(dataset_name))
