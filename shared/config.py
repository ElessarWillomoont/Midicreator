CHECK_POINT = "model_output/archive/ckpt_loss_not_change.pt"
FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes
BATCH_SIZE = 32
PAD_ID = 0  # Assuming 0 is the ID for PAD token
PROJECT_NAME = 'Midicreator_Hugging_face_NO_BPE_smalldata'
ENTITY_NAME = 'candle2587_team'
EPOCH_NUM = 4000
STEP_SIZE = 1000
BATCH_SIZE = 512
MAX_LENGTH = 32
PAD_ID = 0
CHECK_POINT = 'NO'  # Specify your checkpoint path
MAX_LENGTH = 32
VOCAB_SIZE = 465
DECODER_LAYER = 6
HEAD_NUM = 4
N_EMB = 768

vocab_size=465, decoder_layer=6, n_head=4, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID