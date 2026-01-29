from transformers import AutoTokenizer
from config import config

# Load tokenizer globally
tokenizer = AutoTokenizer.from_pretrained(config["decoder_config"]["decoder_model_path"])

# Hàm mã hóa text thành token ids với padding và max_length
def get_caption_tokens(caption_text, max_length=30):
    return tokenizer(
        caption_text, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids[0].tolist()