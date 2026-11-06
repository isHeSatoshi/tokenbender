from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import NEW_INTENTS, SEED
from .io import atomic_bytes, canonical, read_json


def splitmix64(seed: int, count: int) -> np.ndarray:
    if count < 0 or not 0 <= seed < 2**64:
        raise ValueError("Invalid PRNG input")
    with np.errstate(over="ignore"):
        state = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        state += np.uint64(seed)
        z = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return np.asarray(value / norm if norm else np.zeros_like(value), dtype=np.float32)


@dataclass(frozen=True)
class MemoryConfig:
    input_dim: int = 1536
    dimensions: int = 256
    seed: int = SEED
    layer: int = 14
    neighbors: int = 3
    radius: float = 0.1
    blend: float = 0.75
    temperature: float = 0.05
    capacity: int = 2000

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (
                self.input_dim,
                self.dimensions,
                self.seed,
                self.layer,
                self.neighbors,
                self.capacity,
            )
        ) or any(
            type(v) not in (int, float) or not np.isfinite(v)
            for v in (self.radius, self.blend, self.temperature)
        ):
            raise ValueError("Invalid memory configuration types or non-finite settings")
        if (
            self.input_dim < 1
            or self.dimensions < 1
            or self.neighbors < 1
            or not 0 <= self.radius <= 2
            or not 0 <= self.blend <= 1
            or self.temperature <= 0
            or self.capacity < 0
            or self.layer < 0
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("Invalid memory configuration")


class FeedbackMemory:
    def __init__(self, labels: list[str], config: MemoryConfig | None = None):
        if (
            not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
            or len(labels) > 65535
        ):
            raise ValueError("Invalid label vocabulary")
        self.labels = list(labels)
        self.config = config or MemoryConfig()
        bits = splitmix64(self.config.seed, self.config.input_dim * self.config.dimensions)
        signs = (bits & np.uint64(1)).astype(np.float32) * 2 - 1
        self.projection = signs.reshape(self.config.input_dim, self.config.dimensions)
        self.projection /= np.sqrt(self.config.dimensions)
        self.projection_sha256 = hashlib.sha256(self.projection.astype("<f4").tobytes()).hexdigest()
        self.keys: list[np.ndarray] = []
        self.scales: list[float] = []
        self.label_ids: list[int] = []
        self.source_ids: list[str] = []
        self.fingerprints: list[str] = []
        self.quarantined: list[str] = []

    def projected(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.shape != (self.config.input_dim,) or not np.isfinite(hidden).all():
            raise ValueError("Invalid hidden-state key")
        return unit(unit(hidden) @ self.projection)

    def add(self, source_id: str, text: str, label: str, hidden: np.ndarray) -> str:
        if not isinstance(source_id, str) or not source_id or label not in self.labels:
            raise ValueError("Unknown feedback label or invalid source ID")
        if source_id in self.quarantined:
            return "quarantined"
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        if source_id in self.source_ids:
            index = self.source_ids.index(source_id)
            if (
                self.fingerprints[index] != fingerprint
                or self.labels[self.label_ids[index]] != label
            ):
                raise ValueError("A feedback source ID was reused with different content")
            return "duplicate"
        if fingerprint in self.fingerprints:
            index = self.fingerprints.index(fingerprint)
            if self.labels[self.label_ids[index]] != label:
                self.quarantined.append(source_id)
                return "quarantined"
            return "duplicate"
        if len(self.keys) >= self.config.capacity:
            raise OverflowError("Feedback memory is full; no unregistered eviction")
        key = self.projected(hidden)
        scale = np.float32(np.max(np.abs(key)) / 127) if np.any(key) else np.float32(1)
        quantized = np.clip(np.rint(key / scale), -127, 127).astype(np.int8)
        self.keys.append(quantized)
        self.scales.append(float(scale))
        self.label_ids.append(self.labels.index(label))
        self.source_ids.append(source_id)
        self.fingerprints.append(fingerprint)
        return "added"

    def nearest(self, hidden: np.ndarray, count: int | None = None):
        if not self.keys:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError("Invalid neighbor count")
        query = self.projected(hidden)
        keys = (
            np.stack(self.keys).astype(np.float32)
            * np.asarray(self.scales, dtype=np.float32)[:, None]
        )
        norms = np.linalg.norm(keys.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
        keys = np.divide(keys, norms, out=np.zeros_like(keys), where=norms != 0)
        distances = np.clip(1 - keys @ query, 0, 2)
        order = np.argsort(distances, kind="stable")
        if count is not None:
            order = order[:count]
        return order, distances[order]

    def distribution(self, hidden: np.ndarray) -> np.ndarray | None:
        order, distances = self.nearest(hidden, self.config.neighbors)
        admitted = distances <= self.config.radius
        if not admitted.any():
            return None
        order, d = order[admitted], distances[admitted]
        weights = np.exp(-(d - d.min()) / self.config.temperature)
        probabilities = np.zeros(len(self.labels), dtype=np.float64)
        for index, weight in zip(order, weights, strict=True):
            probabilities[self.label_ids[index]] += weight
        return probabilities / probabilities.sum()

    def blend(self, hidden: np.ndarray, model_probabilities: np.ndarray) -> np.ndarray | None:
        memory = self.distribution(hidden)
        if memory is None:
            return None
        probabilities = np.asarray(model_probabilities, dtype=np.float64)
        if (
            probabilities.shape != memory.shape
            or np.any(probabilities < 0)
            or not np.isfinite(probabilities).all()
            or not np.isclose(probabilities.sum(), 1)
        ):
            raise ValueError("Invalid model label probabilities")
        return (1 - self.config.blend) * probabilities + self.config.blend * memory

    @property
    def numeric_entries(self):
        return len(self.keys) * (self.config.dimensions + 2) + len(asdict(self.config))

    def clone(self):
        return copy.deepcopy(self)

    def payload(self):
        # IDs/hashes support replay only; no exemplar text is stored.
        return {
            "format": 1,
            "labels": self.labels,
            "config": asdict(self.config),
            "projection_sha256": self.projection_sha256,
            "keys": [key.tolist() for key in self.keys],
            "scales": self.scales,
            "label_ids": self.label_ids,
            "source_ids": self.source_ids,
            "fingerprints": self.fingerprints,
            "quarantined": self.quarantined,
        }

    def save(self, path: Path):
        atomic_bytes(path, canonical(self.payload()))

    @classmethod
    def load(cls, path: Path):
        return cls.from_payload(read_json(path))

    @classmethod
    def from_payload(cls, value: dict):
        if value["format"] != 1:
            raise ValueError("Unsupported feedback memory")
        memory = cls(value["labels"], MemoryConfig(**value["config"]))
        if memory.projection_sha256 != value["projection_sha256"]:
            raise ValueError("Projection algorithm or seed mismatch")
        for field in ("scales", "label_ids", "source_ids", "fingerprints", "quarantined"):
            setattr(memory, field, value[field])
        raw_keys = np.asarray(value["keys"])
        if raw_keys.size and (
            raw_keys.dtype.kind not in "iu" or (raw_keys < -127).any() or (raw_keys > 127).any()
        ):
            raise ValueError("Invalid int8 key")
        memory.keys = [key.astype(np.int8) for key in raw_keys]
        n = len(memory.keys)
        if (
            n > memory.config.capacity
            or any(
                len(getattr(memory, field)) != n
                for field in ("scales", "label_ids", "source_ids", "fingerprints")
            )
            or any(key.shape != (memory.config.dimensions,) for key in memory.keys)
            or any(
                type(s) not in (float, int)
                or not np.isfinite(s)
                or s <= 0
                or float(np.float32(s)) != s
                for s in memory.scales
            )
            or any(type(i) is not int or not 0 <= i < len(memory.labels) for i in memory.label_ids)
            or any(
                not isinstance(s, str) or not s
                for s in memory.source_ids + memory.fingerprints + memory.quarantined
            )
            or any(
                len(set(getattr(memory, field))) != len(getattr(memory, field))
                for field in ("source_ids", "fingerprints", "quarantined")
            )
            or set(memory.source_ids) & set(memory.quarantined)
        ):
            raise ValueError("Corrupt memory payload")
        return memory


class MemoryRunner:
    def __init__(self, base_runner, memory: FeedbackMemory, mode: str = "constrained"):
        self.base_runner, self.memory, self.mode = base_runner, memory, mode
        self.labels = memory.labels
        self.device = getattr(base_runner, "device", "cpu")
        if self.labels[: len(base_runner.labels)] != base_runner.labels:
            raise ValueError("Memory labels must extend original label IDs")

    def predict(self, text: str, mode: str | None = None) -> str:
        hidden = self.base_runner.feature(text, self.memory.config.layer)
        vote = self.memory.distribution(hidden)
        if vote is None:
            return self.base_runner.predict(text, mode=self.mode)
        if self.memory.config.blend == 1:
            return self.labels[int(vote.argmax())]
        scores = self.base_runner.label_logprobs(text).numpy()
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        full = np.zeros(len(self.labels))
        full[: len(scores)] = probabilities
        blended = (1 - self.memory.config.blend) * full + self.memory.config.blend * vote
        return self.labels[int(blended.argmax())]


def build_stream(teaching: list[dict], corrections: list[dict]) -> list[dict]:
    by_label = {label: [r for r in teaching if r["label"] == label] for label in NEW_INTENTS}
    if any(len(rows) != 10 for rows in by_label.values()):
        raise ValueError("Exactly ten nested teaching examples per new intent are required")
    events = []
    for shot in range(10):
        if shot == 1:
            events.extend(
                {**r, "kind": "correction"} for r in sorted(corrections, key=lambda r: r["id"])
            )
        events.extend(
            {**by_label[label][shot], "kind": "new", "shot": shot + 1} for label in NEW_INTENTS
        )
    return events
