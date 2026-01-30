from transformers import AutoTokenizer

tokenizer = None

def get_tokenizer(model_path="deepseek-ai/deepseek-vl2-tiny"):
    global tokenizer
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Set pad_token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"[get_tokenizer] Failed to load {model_path}: {e}")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_caption_tokens(caption_text, max_length=30, model_path="deepseek-ai/deepseek-vl2-tiny"):
    tok = get_tokenizer(model_path)
    return tok(
        caption_text, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids[0]