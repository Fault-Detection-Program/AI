import os

# Path
PATH = {
    'WEIGHTS_PATH' : None,
    'LABELS' : None
}

# Model Parameter
MODEL = {
    'MODEL_NAME' : None,
    'NUM_CLASSES' : None,
    'LR' : None,
    'MOMENTUM' : None,
    'NUM_EPOCHS' : None,
    'BATCH_SIZE' : None,
    'RESUME_EPOCH' : None
}

def set_config(path):
    global PATH
    global MODEL

    PATH['WEIGHTS_PATH'] = f'{path}/weights'
    PATH['LABELS'] = f'{path}/labels_map.txt'

    MODEL['MODEL_NAME'] = 'efficientnet-b3'
    MODEL['NUM_CLASSES'] = 3
    MODEL['LR'] = 0.01
    MODEL['MOMENTUM'] = 0.9
    MODEL['NUM_EPOCHS'] = 1
    MODEL['BATCH_SIZE'] = 2
    MODEL['RESUME_EPOCH'] = 0