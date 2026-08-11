import logging
import uuid
import torch
import os
from typing import Any, Optional
from prompt_toolkit import PromptSession
import rich
from rich.markdown import Markdown
from rich.panel import Panel
from dataclasses import dataclass, field

from config import load_gemma_config
from chat import Gemma3ChatModel

log = logging.getLogger(__name__)

class ChatPromptLoop:
    def __init__(self, chat_session: "ChatSession", commands: list[Any] | None = None):
        self.chat_session = chat_session
        self.commands = commands or []
        self.prompt_session = PromptSession()

    def initialize(self) -> None:
        with self.chat_session.console.status("Initializing Gemma 3n Chat Model..."):
            self.chat_session.initialize_model()

    def run(self) -> None:
        self.print_welcome()
        while True:
            try:
                user_input = self.prompt_session.prompt("Gemma3n> ")
                if not user_input.strip():
                    continue
                
                if user_input.startswith("/image "):
                    img_path = user_input.split(" ", 1)[1].strip()
                    self.chat_session.load_image(img_path)
                    continue
                    
                if user_input.strip() == "/clear":
                    self.chat_session.history = []
                    self.chat_session.current_image = None
                    self.chat_session.console.print("[blue]Chat cleared.[/blue]")
                    continue

            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat...")
                break

            resp = self.chat_session.chat(user_input)
            self.chat_session.print_response(resp)

    def print_welcome(self):
        welcome_msg = Panel(
            Markdown("# Welcome to Gemma 3n Chat\nMultimodal capabilities enabled (Vision/Audio)."),
            title="Gemma 3n CLI",
            border_style="green"
        )
        self.chat_session.console.print(welcome_msg)

@dataclass
class ChatSession:
    model_path: str = "/run/media/leigh/MEDIA/gemma_3n/"
    config_path: str = "config.json" # Or actual config.json
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    model: Optional[Gemma3ChatModel] = None
    console: rich.console.Console = field(default_factory=rich.console.Console)
    history: list[dict] = field(default_factory=list)
    current_image: Optional[torch.Tensor] = None

    def __post_init__(self):
        if not os.path.exists(self.model_path):
            self.console.print(f"[yellow]Warning: Model path {self.model_path} not found. Using current directory.[/yellow]")
            self.model_path = "."

    def initialize_model(self):
        try:
            from config import Gemma3nConfig
            cfg_file = os.path.join(self.model_path, "config.json")
            if os.path.exists(cfg_file):
                config = load_gemma_config(cfg_file)
            else:
                from config import TextConfig, VisionConfig, AudioConfig
                text_cfg = TextConfig(num_hidden_layers=30) 
                config = Gemma3nConfig(
                    architectures=["Gemma3nForCausalLM"],
                    text=text_cfg,
                    vision=VisionConfig(),
                    audio=AudioConfig()
                )

            self.model = Gemma3ChatModel.from_hf_pretrained(
                model_path=self.model_path,
                config=config,
                device=self.device
            )
            self.model.eval()
        except Exception as e:
            self.console.print(f"[red]Failed to initialize model: {e}[/red]")
            raise e

    def load_image(self, path: str):
        try:
            from PIL import Image
            import torchvision.transforms.functional as TF
            from preprocess import preprocess_image, patchify_image
            
            img = Image.open(path).convert("RGB")
            img_tensor = TF.to_tensor(img) # [C, H, W]
            
            # Preprocess to 896x896 (example)
            img_proc = preprocess_image(img_tensor, image_shape=(896, 896))
            # Patchify
            patches = patchify_image(img_proc, patch_size=14)
            # Add batch and frame dims: [1, 1, num_patches, patch_dim]
            self.current_image = patches.unsqueeze(0).unsqueeze(0).to(self.device)
            self.console.print(f"[green]Loaded image: {path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error loading image: {e}[/red]")

    def chat(self, user_input: str) -> str:
        if self.model is None or self.model.tokenizer is None:
            return "Model or tokenizer not initialized."

        # 1. Construct the formatted prompt with history
        # Template: <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
        
        # Build history string
        prompt_text = ""
        for turn in self.history:
            role = turn["role"]
            content = turn["content"]
            # Map role names if needed (e.g. assistant -> model)
            role_map = {"user": "user", "assistant": "model"}
            mapped_role = role_map.get(role, role)
            prompt_text += f"<start_of_turn>{mapped_role}\n{content}<end_of_turn>\n"

        # Add current user input
        image_marker = chr(65535) if self.current_image is not None else ""
        prompt_text += f"<start_of_turn>user\n{image_marker}{user_input}<end_of_turn>\n"
        prompt_text += f"<start_of_turn>model\n"

        # 2. Tokenize
        # We manually handle BOS to be sure it's at the very start
        tokens = self.model.tokenizer.encode(prompt_text, add_bos=False)
        
        # Prepend BOS ID
        bos_id = getattr(self.model.tokenizer, "bos_id", lambda: 2)()
        if isinstance(bos_id, int):
            tokens = [bos_id] + tokens
        
        # 3. Replace our marker with the actual placeholder value
        from blocks import IMAGE_SOFT_TOKEN_PLACEHOLDER
        processed_tokens = []
        marker_id = self.model.tokenizer.piece_to_id(chr(65535))
        for t in tokens:
            if t == marker_id:
                 processed_tokens.append(IMAGE_SOFT_TOKEN_PLACEHOLDER)
            else:
                 processed_tokens.append(t)
        
        token_tensor = torch.tensor([processed_tokens], device=self.device)
        
        # 4. Generate
        with torch.no_grad():
            try:
                # Stop at either <eos> (1) or <end_of_turn> (107)
                # We'll use the default for now but could pass both if generate supported it
                output_ids = self.model.generate(
                    tokens=token_tensor,
                    images=self.current_image,
                    max_new_tokens=512,
                    eos_token_id=self.model.tokenizer.piece_to_id("<end_of_turn>") or 107
                )
                response_text = self.model.tokenizer.decode(output_ids[0].tolist())
                # Clean up response: remove potential leftover template pieces if any
                if "<end_of_turn>" in response_text:
                    response_text = response_text.split("<end_of_turn>")[0]
            except Exception as e:
                response_text = f"Error during generation: {e}"

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def print_response(self, response: str) -> None:
        output = Markdown(response)
        self.console.print(Panel(output, title="Gemma 3n", border_style="blue"))

if __name__ == "__main__":
    session = ChatSession()
    loop = ChatPromptLoop(session)
    loop.initialize()
    loop.run()
