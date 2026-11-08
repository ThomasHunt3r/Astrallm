
import torch

from tokenizers import Tokenizer

from config import *
from model import AstralLM


class AstralLMChat:

    def __init__(self):

        if not CHECKPOINT_FILE.exists():
            raise FileNotFoundError(
                "Run py train.py to train astralLM 0.2 first."
            )

        if not TOKENIZER_FILE.exists():
            raise FileNotFoundError(
                "BPE tokenizer not found. Run py train.py first."
            )

        print("Loading astralLM 0.2...")

        checkpoint = torch.load(
            CHECKPOINT_FILE,
            map_location=DEVICE,
            weights_only=True
        )

        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_FILE)
        )

        self.model = AstralLM(
            checkpoint["vocab_size"],
            checkpoint["config"]
        ).to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state"]
        )

        self.model.eval()

        self.history = []

        self.user_id = self.tokenizer.token_to_id(
            "<|user|>"
        )

        self.assistant_id = self.tokenizer.token_to_id(
            "<|assistant|>"
        )

        self.eos_id = self.tokenizer.token_to_id(
            "<|eos|>"
        )

        if None in [
            self.user_id,
            self.assistant_id,
            self.eos_id
        ]:
            raise ValueError(
                "Required special tokens are missing."
            )

        print("astralLM 0.2 loaded successfully!")

    # ========================================
    # GENERATE RESPONSE
    # ========================================

    def ask(self, message):

        message = str(message).strip()

        if not message:
            return "Please enter a message."

        # Construct recent conversation
        conversation = []

        for user_message, bot_message in self.history[-3:]:

            conversation.append(
                f"<|user|>{user_message}\n"
                f"<|assistant|>{bot_message}"
                "<|eos|>"
            )

        # Current prompt
        conversation.append(
            f"<|user|>{message}\n<|assistant|>"
        )

        prompt = "\n".join(conversation)

        encoded = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            output = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids=[
                    self.eos_id,
                    self.user_id,
                    self.assistant_id
                ]
            )

        generated = output[0, input_ids.shape[1]:]

        response = self.tokenizer.decode(
            generated.tolist(),
            skip_special_tokens=True
        ).strip()

        if not response:
            response = "I'm still learning. Could you rephrase that?"

        self.history.append(
            (message, response)
        )

        # Preserve recent context
        self.history = self.history[-5:]

        return response

    # ========================================
    # CLEAR MEMORY
    # ========================================

    def clear_memory(self):

        self.history.clear()

        print("Conversation history cleared.")

    # ========================================
    # CHAT INTERFACE
    # ========================================

    def chat(self):

        print("\n" + "=" * 50)
        print("             astralLM 0.2")
        print("=" * 50)

        print("Independent neural language model.")
        print("Type /help for commands.")
        print("Type exit to quit.")

        while True:

            try:

                message = input("\nYou: ").strip()

            except (EOFError, KeyboardInterrupt, OSError):

                print("\nastralLM: Goodbye!")
                break

            if not message:
                continue

            command = message.lower()

            if command in ["exit", "quit"]:

                print("\nastralLM: Goodbye!")
                break

            if command == "/help":

                print("\nCommands:")
                print("/help  - Show commands")
                print("/clear - Clear conversation memory")
                print("exit   - Close astralLM")

                continue

            if command == "/clear":

                self.clear_memory()

                continue

            try:

                response = self.ask(message)

                print("\nastralLM:", response)

            except Exception as error:

                print("\nGeneration error:", error)


def main():

    assistant = AstralLMChat()

    assistant.chat()


if __name__ == "__main__":
    main()
