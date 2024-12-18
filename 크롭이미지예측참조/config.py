import os

# Path
PATH = {
    'DATA_PATH' : None,
    'IMAGE_PATH' : None,
    'INPUT_PATH' : None,
    'OUTPUT_PATH' : None,
    'WEIGHTS_PATH' : None,
    'LABELS' : None,
    'Log' : None
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

def set_config(path_val):
    global PATH
    global MODEL

    PATH['DATA_PATH'] = f'{path_val}/dataset'
    PATH['IMAGE_PATH'] = f'{path_val}/dataset/processed'
    PATH['INPUT_PATH'] = f'{path_val}/input'
    PATH['OUTPUT_PATH'] = f'{path_val}/output'
    PATH['WEIGHTS_PATH'] = f'{path_val}/weights'
    PATH['LABELS'] = f'{path_val}/labels_map.txt'
    PATH['Log'] = f'{path_val}/log'

    MODEL['MODEL_NAME'] = 'efficientnet-b3'
    MODEL['NUM_CLASSES'] = 3
    MODEL['LR'] = 0.01
    MODEL['MOMENTUM'] = 0.9
    MODEL['NUM_EPOCHS'] = 1
    MODEL['BATCH_SIZE'] = 2
    MODEL['RESUME_EPOCH'] = 0