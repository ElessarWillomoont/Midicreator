# Model and training configurations
CHECK_POINT_CONTINUE = "model_output/archive/ckpt_loss_not_change.pt"  # Checkpoint at the beginning of the training
CHECK_POINT_USE = "shared/ckpt/ckpt_pretrained.pt"  # Checkpoint used to generate music
EPOCH_NUM = 4000  # Stop the training when a certain epoch is reached
STEP_SIZE = 1000  # Save checkpoint according to step
BATCH_SIZE = 512  # Adjust this parameter according to your VRAM
MAX_LENGTH = 32  # The context window of the model
PAD_ID = 0  # Assuming 0 is the ID for the PAD token, normally 0, check vocabulary to be sure
VOCAB_SIZE = 465  # Vocabulary size of the model, change this according to your model's vocabulary
DECODER_LAYER = 6  # Number of decoder layers in the model
HEAD_NUM = 4  # Number of attention heads in the model
N_EMB = 768  # Number of embedding dimensions in the model

# Randomness control parameters
TEMPERATURE = 0.9  # Controls randomness, lower values imply less randomness
TOP_K = 50  # Top-K sampling
TOP_P = 0.95  # Nucleus sampling
TARGET_LENGTH = 128  # Target length for music output

# UI window dimensions
WIDTH = 2880  # Window width of the monitor
HEIGHT = 1680  # Window height of the monitor

# File and dataset handling
FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes
PICK_RATIO = 0.1  # Ratio of files to pick for training/validation/evaluation
SIZE_OF_TRAIN = 0.95  # Proportion of the dataset to use for training
SIZE_OF_VAL = 0.04  # Proportion of the dataset to use for validation
SIZE_OF_EVAL = 0.01  # Proportion of the dataset to use for evaluation
# Ensure SIZE_OF_TRAIN + SIZE_OF_VAL + SIZE_OF_EVAL = 1

# User information
PROJECT_NAME = 'your project name'
ENTITY_NAME = 'your username'

# Model parameters (reiterating some values for clarity)
vocab_size = VOCAB_SIZE
decoder_layer = DECODER_LAYER
n_head = HEAD_NUM
n_emb = N_EMB
context_length = MAX_LENGTH
pad_token_id = PAD_ID