from trainers.image_trainer import ImageTrainer
from trainers.text_to_image_trainer import TextToImageTrainer


TRAINERS_DICT = {
    "birds": "ImageTrainer",
    "birds-with-text": "TextToImageTrainer",
    "flowers": "ImageTrainer"
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
        raise Exception('Invalid dataset name {}.'.format(dataset_name))
