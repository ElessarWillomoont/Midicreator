CHECK_POINT = "model_output/archive/ckpt_loss_not_change.pt"
MAX_LENGTH = 32
PAD_ID = 0
MAX_LENGTH = 32  # Maximum sequence length
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

temperature = 0.8  # 控制随机性，较低的值意味着较少的随机性
top_k = 50  # Top-K 抽样
top_p = 0.95  # Nucleus 抽样

tokenizer_path = Path('tokenizer/tokenizer.json')

vocab_size=465, decoder_layer=6, n_head=4, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID