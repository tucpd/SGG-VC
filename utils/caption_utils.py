from transformers import AutoTokenizer

tokenizer = None

def get_tokenizer(model_path="gpt2"):
    global tokenizer
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer

def get_caption_tokens(caption_text, max_length=30, model_path="gpt2"):
    tok = get_tokenizer(model_path)
    return tok(
        caption_text, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).input_ids[0]