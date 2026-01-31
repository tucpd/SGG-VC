import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CaptionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # QFormer da proj roi, chi can proj neu LLM embed_dim khac proj_out_dim
        # DeepSeek-VL2-tiny: hidden_size=1280
        lm_embed_dim = config.get("lm_embed_dim", 1280)  # Default DeepSeek-VL2-tiny
        self.proj = nn.Linear(config["proj_out_dim"], lm_embed_dim)

        # Load pretrained model + tokenizer
        decoder_path = config["decoder_model_path"]
        
        # Try to load DeepSeek-VL2 first
        if "deepseek" in decoder_path.lower():
            try:
                from deepseek_vl2.models import DeepseekVLV2ForCausalLM
                from transformers import AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)
                decoder = DeepseekVLV2ForCausalLM.from_pretrained(
                    decoder_path, 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if config.get("use_fp16", False) else torch.float32
                )
                self.lm_decoder = decoder.language.to(self.device)  # Get language model part only
                print(f"[CaptionHead] Loaded DeepSeek-VL2 from {decoder_path}")
            except Exception as e:
                print(f"Warning: Cannot load DeepSeek-VL2: {e}")
                print("Falling back to standard AutoModelForCausalLM...")
                self._load_auto_model(decoder_path, config)
        else:
            self._load_auto_model(decoder_path, config)
        
        # Set pad_token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for p in self.lm_decoder.parameters():
            p.requires_grad = False

        self.to(self.device)
    
    def _load_auto_model(self, decoder_path, config):
        """Load model using AutoModelForCausalLM"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Warning: Using GPT2 tokenizer as fallback")
        
        try:
            decoder = AutoModelForCausalLM.from_pretrained(decoder_path, trust_remote_code=True)
            if config.get("use_fp16", False):
                decoder = decoder.to(torch.bfloat16)
            self.lm_decoder = decoder.to(self.device)
        except Exception as e:
            print(f"Warning: Cannot load {decoder_path}: {e}")
            from transformers import GPT2LMHeadModel
            self.lm_decoder = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

    def forward(self, qformer_output, truth_caption=None,
                mode='training', inference_strategy='greedy',
                temperature=1.0, top_k=30, top_p=0.9):

        q_feat_proj = self.proj(qformer_output)
        encoder_attention_mask = torch.ones(q_feat_proj.shape[:2], dtype=torch.long).to(self.device)

        if mode == 'training':
            assert truth_caption is not None
            B, T = q_feat_proj.shape[:2]

            # Use same instruction prompt as inference for consistency
            instruction = "Describe this video in one sentence:"
            prompt_inputs = self.tokenizer([instruction] * B, return_tensors="pt", padding=True).to(self.device)
            prompt_embeds = self.lm_decoder.get_input_embeddings()(prompt_inputs["input_ids"])
            
            if isinstance(truth_caption, list) and isinstance(truth_caption[0], str):
                # List of strings
                caption_inputs = self.tokenizer(truth_caption, padding=True, truncation=True, return_tensors="pt").to(self.device)
                caption_embeds = self.lm_decoder.get_input_embeddings()(caption_inputs["input_ids"])
                truth_caption_ids = caption_inputs["input_ids"]
            elif isinstance(truth_caption, list) and torch.is_tensor(truth_caption[0]):
                # List of tensors - stack them
                truth_caption = torch.stack(truth_caption, dim=0).to(self.device)
                caption_embeds = self.lm_decoder.get_input_embeddings()(truth_caption)
                truth_caption_ids = truth_caption
            else:
                # Single tensor
                caption_embeds = self.lm_decoder.get_input_embeddings()(truth_caption.to(self.device))
                truth_caption_ids = truth_caption.to(self.device)

            combined_embeds = torch.cat([
                q_feat_proj.to(prompt_embeds.dtype).contiguous(),
                prompt_embeds,
                caption_embeds
            ], dim=1)

            attention_mask = torch.cat([
                torch.ones((B, T), dtype=torch.long).to(self.device),
                prompt_inputs["attention_mask"].to(self.device),
                torch.ones((B, caption_embeds.shape[1]), dtype=torch.long).to(self.device)
            ], dim=1).to(self.device)

            labels = torch.cat([
                torch.full((B, T + prompt_embeds.shape[1]), -100, dtype=torch.long).to(self.device),
                truth_caption_ids
            ], dim=1).to(self.device)

            outputs = self.lm_decoder(
                attention_mask=attention_mask,
                inputs_embeds=combined_embeds,
                labels=labels,
                return_dict=True,
            )
            return outputs.loss, outputs.logits, q_feat_proj

        elif mode == 'inference':
            B, T = q_feat_proj.shape[:2]
            
            # Use instruction prompt to guide caption generation
            instruction = "Describe this video in one sentence:"
            prompt_inputs = self.tokenizer([instruction] * B, return_tensors="pt", padding=True).to(self.device)
            prompt_embeds = self.lm_decoder.get_input_embeddings()(prompt_inputs["input_ids"])

            # Visual features + instruction prompt
            combined_embeds = torch.cat([q_feat_proj.to(prompt_embeds.dtype), prompt_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones((B, T), dtype=torch.long).to(self.device),
                prompt_inputs["attention_mask"].to(self.device)
            ], dim=1).to(self.device)

            generated = self.lm_decoder.generate(
                attention_mask=attention_mask,
                inputs_embeds=combined_embeds,
                do_sample=False,
                num_beams=self.config.get("num_beams", 5),
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.get("max_new_tokens", 32),
                min_new_tokens=5,
                repetition_penalty=1.2,
            )
            return generated

        else:
            raise ValueError(f"Unknown mode: {mode}")
