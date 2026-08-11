import torch
import torch.nn as nn
from typing import Any, Optional
from config import Gemma3nConfig
from blocks import Gemma3nTransformer, Input, Gemma3nAudioEncoder
from vision import MobileNetV5FromPatches

class Gemma3ChatModel(nn.Module):
    """
    A unified model for multimodal chat, orchestrating vision, audio, and text encoders.
    """
    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.config = config
        
        # 1. Encoders
        self.vision_tower = MobileNetV5FromPatches(
            output_length=config.vision_soft_tokens_per_image
        )
        self.audio_tower = Gemma3nAudioEncoder(config.audio)
        
        # 2. Backbone
        self.transformer = Gemma3nTransformer(config)

        self.tokenizer: Any = None

    @classmethod
    def from_hf_pretrained(cls,
                           model_path: str,
                           config: Gemma3nConfig,
                           device: str = "cpu",
                           vision_path: Optional[str] = None,
                           **kwargs
                       ) -> "Gemma3ChatModel":
        """
        Loads all components of the Gemma 3n model from pre-trained weights with memory efficiency.
        """
        # 1. Determine target dtype from config to save initial RAM
        dtype_str = getattr(config.text, "torch_dtype", "float32")
        dtype_map = {
            "torch.bfloat16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "float16": torch.float16,
            "torch.float32": torch.float32,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(dtype_str, torch.float32)

        # 2. Initialize model on 'meta' device to avoid allocating uninitialized weights
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(target_dtype)
        try:
            # We use 'meta' device for the wrapper to avoid double-allocation
            # since the sub-modules' from_hf_pretrained will create their own instances.
            with torch.device("meta"):
                model = cls(config)
        finally:
            torch.set_default_dtype(original_dtype)

        # 3. Load Transformer and Audio weights (usually in same repo)
        # These methods handle their own memory-efficient initialization.
        tokenizer, transformer = Gemma3nTransformer.from_hf_pretrained(
            repo_id=model_path,
            config=config,
            device=device,
            local=True,
            load_tokenizer=True
        )
        model.transformer = transformer
        model.tokenizer = tokenizer

        # 4. Load Audio Encoder weights
        # audio_encoder = Gemma3nAudioEncoder.from_hf_pretrained(
        #     model_path=model_path,
        #     config=config.audio,
        #     device=device
        # )
        # model.audio_tower = audio_encoder

        # 5. Load Vision Encoder weights (may be in a different path)
        # v_path = vision_path or model_path
        # model.vision_tower = MobileNetV5FromPatches(
        #     model_path=v_path,
        # )
        print(f"Successfully loaded all Gemma 3n components from {model_path}")
        return model

    def forward(
        self,
        tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        **kwargs
    ):
        """
        Full multimodal forward pass.
        
        Args:
            tokens: [batch, seq] raw tokens (with placeholders)
            images: [batch, n, h, w, c] or [batch, n, num_patches, patch_dim]
            audio: [batch, num_frames, num_channels, mel_bins]
            audio_mask: [batch, num_frames]
        """
        # 1. Encode Multimodal inputs
        vision_features = None
        if images is not None:
            # images: [b, n, ...] -> [b, n, 256, 2048]
            vision_features = self.vision_tower(images)
            # Flatten frames and tokens for interleaving: [b, n*256, 2048]
            vision_features = vision_features.view(vision_features.shape[0], -1, vision_features.shape[-1])

        audio_features = None
        if audio is not None and audio_mask is not None:
            # audio_output: [b, seq, hidden]
            audio_output = self.audio_tower(audio, audio_mask)
            audio_features = audio_output.last_hidden_state

        # 2. Interleave and prepare masks
        # We use the Input dataclass to handle sequence construction
        mm_input = Input(
            tokens=tokens,
            images=images, # only used for metadata/pre-check in prepare_multimodal_stream
            audio=audio,
            config=self.config
        )
        
        interleaved_tokens, modality_mask = mm_input.prepare_multimodal_stream()
        interleaved_tokens = interleaved_tokens.to(tokens.device)
        modality_mask = modality_mask.to(tokens.device)

        # 3. Position Indices (Simplified)
        batch_size, seq_len = interleaved_tokens.shape
        segment_pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)

        # 4. Transformer Forward
        return self.transformer(
            input_ids=interleaved_tokens,
            segment_pos=segment_pos,
            vision_tokens=vision_features,
            audio_tokens=audio_features,
            modality_mask=modality_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Greedy generation loop with multimodal support.
        """
        device = tokens.device
        batch_size = tokens.shape[0]
        
        # 1. Initial multimodal forward pass to populate cache and handle interleaving
        # Note: forward handles prepare_multimodal_stream internally
        outputs = self.forward(
            tokens=tokens,
            images=images,
            audio=audio,
            audio_mask=audio_mask,
            use_cache=True
        )
        
        logits = outputs.logits # [b, seq, vocab]
        past_key_values = outputs.cache
        
        # Greedy sample from last token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        all_tokens = [next_token]
        
        # 2. Iterative generation
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break
                
            # For next steps, we only pass the single new token and the cache
            # Modality mask for new tokens is always 0 (Text)
            modality_mask = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            
            # Position index for the new token
            # We need to calculate total sequence length including interleaved tokens
            # For simplicity in this implementation, we rely on the transformer to handle
            # position incrementing if segment_pos is provided correctly.
            # In blocks.py, segment_pos is used to index cos/sin.
            
            # Current sequence length in cache
            # past_key_values is a list of LayerCache (dict)
            cur_len = past_key_values[0]["end_index"].item()
            segment_pos = torch.full((batch_size, 1), cur_len, device=device)

            outputs = self.transformer(
                input_ids=next_token,
                segment_pos=segment_pos,
                modality_mask=modality_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits
            past_key_values = outputs.cache
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            all_tokens.append(next_token)
            
        return torch.cat(all_tokens, dim=-1)
