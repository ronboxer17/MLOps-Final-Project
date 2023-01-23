from pathlib import Path
import os

SEED = 11

ROOT_DIR = Path(__file__).parent  # This is the project Root
DATA_FOLDER_PATH = os.path.join(ROOT_DIR, 'data')

FRENCH_PATH = os.path.join(DATA_FOLDER_PATH, 'freMTPL2freq.csv')
BOSTON_PATH = os.path.join(DATA_FOLDER_PATH, 'housing.csv')
RESULTS_PATH = os.path.join(DATA_FOLDER_PATH, 'results.csv')
RESULTS_BOSTON_PATH = os.path.join(DATA_FOLDER_PATH, 'results_boston.csv')
RESULTS_FRENCH_PATH = os.path.join(DATA_FOLDER_PATH, 'results_french.csv')


DATASETS_NAMES_LABELS = {
    'boston_housing': 'MEDV',
    'french_motor_claims': 'ClaimNb'
}

RESULTS_PATH_DICT = {
    'boston_housing': RESULTS_BOSTON_PATH,
    'french_motor_claims': RESULTS_FRENCH_PATH
}