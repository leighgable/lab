import sentencepiece as spm
import os

model_path = "/run/media/leigh/MEDIA/gemma_3n/tokenizer.model"
if os.path.exists(model_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    ids = [2, 105, 2364, 107, 9259, 236888, 106, 107, 105, 4368, 107]
    for i in ids:
        print(f"ID {i:6d}: '{sp.IdToPiece(i)}'")
else:
    print("Tokenizer model not found.")
