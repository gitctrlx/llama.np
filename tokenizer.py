from typing import List, Union
import json
import numpy as np


class Tokenizer:
    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.vocab = model["tokens"]
        self.scores = model["scores"]
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.unk_id = self.str_lookup("<unk>")
        self.bos_id = self.str_lookup("<s>")
        self.eos_id = self.str_lookup("</s>")
        if self.bos_id == -1:
            self.bos_id = 1  # Fallback to hardcoded if not found
        if self.eos_id == -1:
            self.eos_id = 2  # Fallback to hardcoded if not found
        if self.unk_id == -1:
            self.unk_id = 0

    def str_lookup(self, token: str) -> int:
        return self.token_to_id.get(token, -1)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        # Convert to bytes for byte-level BPE
        text_bytes = text.encode("utf-8")
        tokens = [int(b) for b in text_bytes]  # Base tokens are byte values 0-255

        while len(tokens) >= 2:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Build the merged string
                pair_str = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(pair_str)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break  # No more merges possible

            # Merge the pair
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]  # Remove the next token

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: Union[int, np.integer, List[int], np.ndarray]) -> str:
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]
        elif isinstance(ids, np.ndarray):
            ids = ids.flatten().astype(int).tolist()
        res = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                res.append(self.vocab[self.unk_id])  # Use unk for invalid IDs
            else:
                res.append(self.vocab[i])
        text = "".join(res)
        # Do not auto-strip specials; let caller handle if needed
        return text
