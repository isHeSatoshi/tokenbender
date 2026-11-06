from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .constants import SEED, SYSTEM_PROMPT


def deterministic(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_tokenizer(path: Path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def model_config(path: Path):
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    # Older Qwen2 implementations use rope_theta rather than rope_parameters.
    rope = getattr(config, "rope_parameters", None)
    if rope and "rope_theta" in rope and not hasattr(config, "rope_theta"):
        config.rope_theta = rope["rope_theta"]
    return config


def load_base(path: Path, device: str = "cuda"):
    deterministic()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=model_config(path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if (
        model.get_input_embeddings().weight.data_ptr()
        != model.get_output_embeddings().weight.data_ptr()
    ):
        raise ValueError("Expected tied input and output embeddings")
    return model


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def prompt(self, text: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

    def label(self, text: str, label: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = self.prompt(text)
        complete = self.tokenizer.encode(rendered + label, add_special_tokens=False)
        if complete[: len(prompt)] != prompt:
            raise ValueError("Label changes prompt-boundary tokenization")
        return complete[len(prompt) :] + [self.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def original_id(self, token: int) -> int:
        return token


class LabelTrie:
    TERMINAL = -1

    def __init__(self, sequences: list[list[int]], eos: int):
        self.root = {}
        self.eos = eos
        for sequence in sequences:
            if not sequence or sequence[-1] != eos:
                raise ValueError("Each trie path must end in EOS")
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})
            node[self.TERMINAL] = True

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            if token not in node:
                raise ValueError("Generated prefix is not a valid label")
            node = node[token]
        return sorted(token for token in node if token != self.TERMINAL)


class Runner:
    def __init__(self, model, tokenizer: TokenizerAdapter, labels: list[str]):
        self.model, self.tokenizer, self.labels = model, tokenizer, labels
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def predict(self, text: str, mode: str = "greedy", max_new_tokens: int = 20) -> str:
        if mode == "full-label":
            scores = self.label_logprobs(text)
            return self.labels[int(scores.argmax())]
        if mode not in {"greedy", "constrained"}:
            raise ValueError(f"Unknown decoding mode: {mode}")
        prompt = self.tokenizer.prompt(text)
        paths = [self.tokenizer.label(text, label) for label in self.labels]
        if max(map(len, paths)) > max_new_tokens:
            raise ValueError("Label+EOS path exceeds registered token cap")
        trie = LabelTrie(paths, self.tokenizer.eos_token_id) if mode == "constrained" else None
        ids = torch.tensor([prompt], device=self.device)
        generated, past = [], None
        for _ in range(max_new_tokens):
            output = self.model(ids, past_key_values=past, use_cache=True, return_dict=True)
            logits = output.logits[0, -1]
            if trie:
                allowed = sorted(trie.allowed(generated), key=self.tokenizer.original_id)
                if not allowed:
                    raise RuntimeError("Trie ended before EOS")
                choices = torch.tensor(allowed, device=self.device)
                token = int(choices[logits[choices].argmax()])
            else:
                # Compact IDs preserve original-ID ordering.
                token = int(logits.argmax())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                break
            past = output.past_key_values
            ids = torch.tensor([[token]], device=self.device)
        return self.tokenizer.decode(generated)

    @torch.inference_mode()
    def label_logprobs(self, text: str) -> torch.Tensor:
        prompt = self.tokenizer.prompt(text)
        scores = []
        # A single label at a time bounds peak logits memory independently of label count.
        for label in self.labels:
            continuation = self.tokenizer.label(text, label)
            ids = torch.tensor([prompt + continuation[:-1]], device=self.device)
            hidden = self.model.model(ids, use_cache=False, return_dict=True).last_hidden_state
            logits = self.model.lm_head(hidden[:, len(prompt) - 1 :]).float()
            targets = torch.tensor(continuation, device=self.device)
            scores.append(logits[0].log_softmax(-1).gather(1, targets[:, None]).sum())
        return torch.stack(scores).cpu()

    @torch.inference_mode()
    def feature(self, text: str, layer: int) -> np.ndarray:
        ids = torch.tensor([self.tokenizer.prompt(text)], device=self.device)
        output = self.model.model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        if not 0 <= layer < len(output.hidden_states):
            raise ValueError("Feature layer is outside this model")
        return output.hidden_states[layer][0, -1].float().cpu().numpy()
