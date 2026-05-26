# src/config.py

class_names = ['Hello', 'Helpme', 'Stop', 'Thankyou', 'Yes']
NUM_CLASSES = 5

# Routing map (fixed)
LENGTH_MAP  = {0: 1, 1: 1, 2: 0, 3: 1, 4: 0}  # short=0, long=1
SHORT_REMAP = {2: 0, 4: 1}
LONG_REMAP  = {0: 0, 1: 1, 3: 2}

FS = 256
CROP_LEN = 256
PAPER_STRIDE = 32

# User can override these
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 80
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-4

ENSEMBLE_CONFIGS = [
    dict(name="A", seed=42,  crop_stride=PAPER_STRIDE, kernels=(7,15,31)),
    dict(name="B", seed=123, crop_stride=PAPER_STRIDE, kernels=(9,19,39)),
    dict(name="C", seed=999, crop_stride=PAPER_STRIDE, kernels=(5,11,23)),
]

PROBE_MEMBER = "A"
RANDOM_STATE = 42
CALIB_FRAC = 0.20
N_FOLDS = 5

CONF_FILTER_GRID = [False, True]
EMA_MOM = 0.80

# Training defaults
ARC_S = 24.0
ARC_M = 0.35
LAMBDA_ARC = 0.35
ARC_START_EPOCH = 11
MIN_LR_RATIO  = 0.05
WARMUP_EPOCHS = 5

EARLYSTOP_MIN_EPOCH = 15
EARLYSTOP_PATIENCE  = 10
EVAL_EVERY_EPOCHS    = 1
CHEAP_EVAL_MAX_TRIALS = 180
FULL_EVAL_EVERY       = 5
