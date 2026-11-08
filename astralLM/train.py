
import random
import re

import torch

from tokenizers import Tokenizer, models
from tokenizers import trainers, pre_tokenizers, decoders

from config import *
from model import AstralLM


# ============================================
# TEXT PROCESSING
# ============================================

def normalize_chat_format(text):

    lines = []

    for line in text.splitlines():

        line = line.rstrip()

        if re.match(r"^\s*User:", line):
            line = re.sub(
                r"^\s*User:",
                "<|user|>",
                line
            )

        elif re.match(r"^\s*(astralLM|NOVA):", line, re.I):

            line = re.sub(
                r"^\s*(astralLM|NOVA):",
                "<|assistant|>",
                line,
                flags=re.I
            )

            if not line.endswith("<|eos|>"):
                line += " <|eos|>"

        lines.append(line)

    return "\n".join(lines)


def split_corpus(text):

    marker = "==words=="

    if marker in text:
        language_text, word_text = text.split(marker, 1)
    else:
        language_text = text
        word_text = ""

    language_documents = [
        doc.strip()
        for doc in re.split(r"\n\s*\n", language_text)
        if doc.strip()
    ]

    # Group dictionary entries into manageable documents
    word_lines = [
        line.strip()
        for line in word_text.splitlines()
        if line.strip()
    ]

    word_documents = []

    chunk_size = 250

    for i in range(0, len(word_lines), chunk_size):

        chunk = word_lines[i:i + chunk_size]

        word_documents.append(
            "\n".join(chunk)
        )

    return language_documents, word_documents


def split_documents(documents, seed=42):

    documents = list(documents)

    rng = random.Random(seed)

    rng.shuffle(documents)

    if len(documents) == 0:
        return [], []

    if len(documents) == 1:

        doc = documents[0]

        cut = max(1, int(len(doc) * 0.9))

        return [doc[:cut]], [doc[cut:]]

    validation_count = max(
        1,
        int(len(documents) * 0.10)
    )

    validation_documents = documents[:validation_count]
    training_documents = documents[validation_count:]

    return training_documents, validation_documents


# ============================================
# BPE TOKENIZER
# ============================================

def train_tokenizer(text):

    print("\nTraining BPE tokenizer...")

    tokenizer = Tokenizer(
        models.BPE(unk_token="<|unk|>")
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    tokenizer.train_from_iterator(
        lines,
        trainer=trainer,
        length=len(lines)
    )

    tokenizer.save(str(TOKENIZER_FILE))

    print(
        "Tokenizer vocabulary:",
        tokenizer.get_vocab_size()
    )

    print("Tokenizer saved:", TOKENIZER_FILE)

    return tokenizer


# ============================================
# TOKENIZE DOCUMENTS
# ============================================

def encode_documents(documents, tokenizer):

    eos_id = tokenizer.token_to_id("<|eos|>")

    if eos_id is None:
        raise ValueError("EOS token is missing.")

    token_ids = []

    for document in documents:

        encoded = tokenizer.encode(
            document,
            add_special_tokens=False
        )

        token_ids.extend(encoded.ids)

        # Separate documents
        token_ids.append(eos_id)

    if not token_ids:
        return None

    return torch.tensor(
        token_ids,
        dtype=torch.long
    )


# ============================================
# BATCH GENERATION
# ============================================


def get_batch(source):

    if source is None or source.numel() < 2:
        raise ValueError(
            "Not enough tokens to create a training batch."
        )

    sequence_length = min(
        BLOCK_SIZE,
        source.numel() - 1
    )

    # Exclude the final invalid starting position.
    max_start_exclusive = (
        source.numel() - sequence_length
    )

    starts = torch.randint(
        0,
        max_start_exclusive,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[
            int(i):int(i) + sequence_length
        ]
        for i in starts
    ])

    y = torch.stack([
        source[
            int(i) + 1:int(i) + sequence_length + 1
        ]
        for i in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)

def choose_training_stream(streams):

    available = {
        name: stream
        for name, stream in streams.items()
        if stream is not None and stream.numel() >= 3
    }

    if not available:
        raise ValueError("No usable training streams.")

    weights = []

    for name in available:

        if name == "language":
            weights.append(LANGUAGE_WEIGHT)

        else:
            weights.append(WORDLIST_WEIGHT)

    selected = random.choices(
        list(available.keys()),
        weights=weights,
        k=1
    )[0]

    return available[selected]


# ============================================
# VALIDATION
# ============================================

@torch.no_grad()
def evaluate(model, streams):

    model.eval()

    results = {}

    for name, source in streams.items():

        if source is None or source.numel() < 3:
            continue

        losses = []

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(source)

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# ============================================
# MAIN TRAINING
# ============================================

def main():

    random.seed(42)
    torch.manual_seed(42)

    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_FILE}"
        )

    raw_text = CORPUS_FILE.read_text(
        encoding="utf-8"
    )

    if len(raw_text) < 1000:

        raise ValueError(
            "Corpus is too small for this experiment."
        )

    text = normalize_chat_format(raw_text)

    language_docs, word_docs = split_corpus(text)

    language_characters = sum(
        len(doc) for doc in language_docs
    )

    word_characters = sum(
        len(doc) for doc in word_docs
    )

    print("=" * 55)
    print("ASTRALLM 0.2")
    print("=" * 55)

    print("Total corpus characters:", len(raw_text))
    print("Language characters:", language_characters)
    print("Word-list characters:", word_characters)
    print("Language documents:", len(language_docs))
    print("Word-list documents:", len(word_docs))

    if language_characters < 10000:

        print("\nWARNING:")
        print(
            "Your corpus contains very little natural-language "
            "material."
        )
        print(
            "The model can learn language patterns, but it "
            "cannot become broadly knowledgeable from this data."
        )

    # Train tokenizer using the complete single corpus
    tokenizer = train_tokenizer(text)

    # Split natural-language documents
    language_train, language_val = split_documents(
        language_docs,
        seed=42
    )

    # Split word-list documents independently
    word_train, word_val = split_documents(
        word_docs,
        seed=43
    )

    train_streams = {
        "language": encode_documents(
            language_train,
            tokenizer
        )
    }

    validation_streams = {
        "language": encode_documents(
            language_val,
            tokenizer
        )
    }

    if word_train:

        train_streams["words"] = encode_documents(
            word_train,
            tokenizer
        )

        validation_streams["words"] = encode_documents(
            word_val,
            tokenizer
        )

    print("\nTRAINING TOKENS")

    for name, stream in train_streams.items():

        print(
            f"{name}: "
            f"{stream.numel() if stream is not None else 0:,}"
        )

    print("\nVALIDATION TOKENS")

    for name, stream in validation_streams.items():

        print(
            f"{name}: "
            f"{stream.numel() if stream is not None else 0:,}"
        )

    vocab_size = tokenizer.get_vocab_size()

    config = {
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "dropout": DROPOUT
    }

    model = AstralLM(
        vocab_size,
        config
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nMODEL")
    print("Vocabulary:", vocab_size)
    print("Parameters:", f"{parameters:,}")
    print("Device:", DEVICE)

    print("\n" + "=" * 55)
    print("STARTING TRAINING")
    print("=" * 55)

    model.train()

    for iteration in range(MAX_ITERS):

        if iteration % EVAL_INTERVAL == 0:

            results = evaluate(
                model,
                validation_streams
            )

            status = " | ".join(
                f"{name} val: {loss:.4f}"
                for name, loss in results.items()
            )

            print(
                f"Step {iteration}/{MAX_ITERS} | {status}"
            )

        source = choose_training_stream(
            train_streams
        )

        x, y = get_batch(source)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

    # ========================================
    # SAVE CHECKPOINT
    # ========================================

    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size,
        "tokenizer_path": TOKENIZER_FILE.name,
        "version": "0.2"
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE
    )

    print("\n" + "=" * 55)
    print("ASTRALLM 0.2 TRAINING COMPLETE")
    print("=" * 55)

    print("Checkpoint:", CHECKPOINT_FILE)
    print("Tokenizer:", TOKENIZER_FILE)

    # ========================================
    # SAMPLE GENERATION
    # ========================================

    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    prompt = "<|user|>Hello!\n<|assistant|>"

    prompt_ids = torch.tensor(
        [tokenizer.encode(
            prompt,
            add_special_tokens=False
        ).ids],
        dtype=torch.long,
        device=DEVICE
    )

    model.eval()

    output = model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        stop_token_ids=[eos_id, user_id]
    )

    generated_ids = output[0, prompt_ids.shape[1]:]

    print("\nGENERATED RESPONSE:")
    print(
        tokenizer.decode(
            generated_ids.tolist(),
            skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()
