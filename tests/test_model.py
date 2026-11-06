import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from tenpercent.model import Runner


class Adapter:
    eos_token_id = 2

    def prompt(self, text):
        return [1, 3, 4]

    def label(self, text, label):
        return {"short": [5, 2], "long": [6, 7, 8, 2], "prefix": [6, 2]}[label]


def runner():
    torch.manual_seed(4)
    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    return Runner(model, Adapter(), ["short", "long", "prefix"])


def test_batched_label_scores_match_independent_full_model_teacher_forcing():
    r = runner()
    expected = []
    with torch.inference_mode():
        for label in r.labels:
            prompt, answer = r.tokenizer.prompt("x"), r.tokenizer.label("x", label)
            ids = torch.tensor([prompt + answer[:-1]])
            logp = r.model(ids).logits[0, len(prompt) - 1 :].log_softmax(-1)
            expected.append(sum(logp[i, token] for i, token in enumerate(answer)))
    expected = torch.stack(expected)
    for size in (1, 2, 3, 8):
        assert torch.allclose(r.label_logprobs("x", size), expected, atol=2e-6)
    with pytest.raises(ValueError):
        r.label_logprobs("x", 0)


def test_multi_layer_features_use_last_prompt_token_and_do_not_alias():
    r = runner()
    features = r.features("x", [0, 1, 2])
    with torch.inference_mode():
        expected = r.model.model(torch.tensor([[1, 3, 4]]), output_hidden_states=True).hidden_states
    for layer in features:
        assert np.array_equal(features[layer], expected[layer][0, -1].numpy())
        assert np.array_equal(features[layer], r.feature("x", layer))
    with pytest.raises(ValueError):
        r.features("x", [3])
