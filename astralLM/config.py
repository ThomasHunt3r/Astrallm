
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# The ONLY training data file
CORPUS_FILE = DATA_DIR / "corpus.txt"

# New version-specific files
CHECKPOINT_FILE = CHECKPOINT_DIR / "astralLM_02.pt"
TOKENIZER_FILE = CHECKPOINT_DIR / "astralLM_tokenizer.json"

DEVICE = "cpu"

# Transformer
BLOCK_SIZE = 128
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

# BPE tokenizer
BPE_VOCAB_SIZE = 1024
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<|unk|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>"
]

# Training
BATCH_SIZE = 8
MAX_ITERS = 1500
LEARNING_RATE = 0.0003

EVAL_INTERVAL = 250
EVAL_BATCHES = 8

# Generation
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40
REPETITION_PENALTY = 1.1

# Balance natural language and vocabulary samples
LANGUAGE_WEIGHT = 0.75
WORDLIST_WEIGHT = 0.25
