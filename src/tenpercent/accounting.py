from __future__ import annotations

from dataclasses import asdict, dataclass

from .constants import BASE_PARAMETERS, BYTE_CAP, PARAMETER_CAP, SOURCE_TARGET


@dataclass(frozen=True)
class Budget:
    vocabulary_rows: int
    group_size: int = 4
    source_target: int = SOURCE_TARGET
    auxiliary_entries: int = 0
    hidden: int = 1536
    fixed_body_entries: int = 144_896
    body_entries: int = 1_310_195_712
    diagnostic_oversize: bool = False

    def __post_init__(self):
        if (
            self.vocabulary_rows < 1
            or self.group_size not in (2, 4)
            or self.auxiliary_entries < 0
            or self.body_entries % self.group_size
        ):
            raise ValueError("Invalid budget")
        if self.keep_groups < 0 or self.keep_groups > self.total_groups:
            raise ValueError("Vocabulary/fixed values do not fit requested source target")
        cap = BASE_PARAMETERS if self.diagnostic_oversize else PARAMETER_CAP
        if self.source_entries + self.auxiliary_entries > cap:
            raise ValueError("Auxiliary state exceeds the retained-entry cap")

    @property
    def total_groups(self):
        return self.body_entries // self.group_size

    @property
    def dense_entries(self):
        return self.hidden * self.vocabulary_rows + self.fixed_body_entries

    @property
    def keep_groups(self):
        return (self.source_target - self.dense_entries) // self.group_size

    @property
    def source_entries(self):
        return self.dense_entries + self.group_size * self.keep_groups

    def report(self):
        return {
            **asdict(self),
            "keep_groups": self.keep_groups,
            "total_groups": self.total_groups,
            "source_entries": self.source_entries,
            "retained_entries": self.source_entries + self.auxiliary_entries,
            "body_density": self.keep_groups / self.total_groups,
            "base_parameters": BASE_PARAMETERS,
            "parameter_cap": PARAMETER_CAP,
            "byte_cap": BYTE_CAP,
        }


def cubic_group_count(step: int, steps: int, total: int, target: int) -> int:
    if steps < 1 or not 0 <= step <= steps or not 0 <= target <= total:
        raise ValueError("Invalid density schedule")
    progress = step / steps
    u = min(1.0, max(0.0, (progress - 0.1) / 0.7))
    return max(target, min(total, round(target + (total - target) * (1 - u) ** 3)))
