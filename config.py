from enum import Enum


class PredictionType(str, Enum):
    NEXT_POINT = 'Next point prediction'
    MULTIPLE_SEQUENCES = 'Multiple sequence prediction'
    FULL_SEQUENCE = 'Full sequence prediction'


# select prediction type
PREDICTION_TYPE = PredictionType.MULTIPLE_SEQUENCES

COMPANY_TICKER = "^FTSE"
TRAIN_SPLIT = 0.85

# data loader dates
START_DATE = "2000-01-01"
END_DATE = "2023-09-01"


STEPS = 6
LEARNING_RATE = 0.8
HIDDEN_LAYERS = 64
WINDOW_SIZE = 50
