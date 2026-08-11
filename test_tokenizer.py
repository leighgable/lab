import sentencepiece as spm
import os

model_path = "/run/media/leigh/MEDIA/gemma_3n/tokenizer.model"
if os.path.exists(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    text = "Hello!"
    tokens = sp.EncodeAsIds(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    for t in tokens:
        print(f"ID {t}: {sp.IdToPiece(t)}")
else:
    print("Tokenizer model not found.")
