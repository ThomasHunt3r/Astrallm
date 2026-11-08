
from config import CORPUS_FILE, CHECKPOINT_FILE
import torch

print("=" * 50)
print("ASTRALLM DIAGNOSTICS")
print("=" * 50)

# ============================================
# CHECK TRAINING CORPUS
# ============================================

if not CORPUS_FILE.exists():
    raise FileNotFoundError(
        f"Corpus not found: {CORPUS_FILE}"
    )

text = CORPUS_FILE.read_text(encoding="utf-8")

print("\nCORPUS INFORMATION")
print("File:", CORPUS_FILE)
print("Characters:", len(text))
print("Unique characters:", len(set(text)))

print("\nCONVERSATION FORMATS")
print("Contains User:", "User:" in text)
print("Contains astralLM:", "astralLM:" in text)
print("Contains NOVA:", "NOVA:" in text)

print("\nFIRST 1000 CHARACTERS:")
print(text[:1000])

# ============================================
# CHECK MODEL CHECKPOINT
# ============================================

print("\nMODEL INFORMATION")

if not CHECKPOINT_FILE.exists():

    print("No trained checkpoint found.")
    print("Run py train.py first.")

else:

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=True
    )

    print("Checkpoint:", CHECKPOINT_FILE)
    print("Vocabulary:", checkpoint["vocab_size"])
    print("Context length:", checkpoint["config"]["block_size"])
    print("Embedding dimensions:", checkpoint["config"]["n_embd"])
    print("Transformer layers:", checkpoint["config"]["n_layer"])
    print("Attention heads:", checkpoint["config"]["n_head"])

    stoi = checkpoint["stoi"]

    prompt = "User: hi\nastralLM:"

    unknown = sorted(set(prompt) - set(stoi))

    print("\nUNKNOWN CHAT CHARACTERS:")
    print(unknown if unknown else "None")

# ============================================
# COMPLETE
# ============================================

print("\n" + "=" * 50)
print("DIAGNOSTICS COMPLETE")
print("=" * 50)
