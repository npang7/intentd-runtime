"""Typed scenario contracts consumed by both the grader and tests."""

from __future__ import annotations

from dataclasses import dataclass

from sim.network import ActionValue, NetworkState


@dataclass(frozen=True)
class Threshold:
    min_value: float | None = None
    max_value: float | None = None

    def contains(self, value: float) -> bool:
        return (self.min_value is None or value >= self.min_value) and (
            self.max_value is None or value <= self.max_value
        )

    def describe(self) -> str:
        return (
            f"[{self.min_value if self.min_value is not None else '-inf'}, "
            f"{self.max_value if self.max_value is not None else 'inf'}]"
        )


@dataclass(frozen=True)
class MetricRequirement:
    cell_id: str
    metric_name: str


@dataclass(frozen=True)
class ActionMatcher:
    tools: tuple[str, ...]
    cell_id: str | None = None
    category: str | None = None

    def matches(self, action: ActionValue) -> bool:
        if action.tool not in self.tools:
            return False
        if self.cell_id is not None and getattr(action, "cell_id", None) != self.cell_id:
            return False
        return self.category is None or getattr(action, "category", None) == self.category

    def describe(self) -> str:
        details = [f"tool in {self.tools}"]
        if self.cell_id is not None:
            details.append(f"cell_id={self.cell_id}")
        if self.category is not None:
            details.append(f"category={self.category}")
        return ", ".join(details)


@dataclass(frozen=True)
class ActionRequirement:
    matchers: tuple[ActionMatcher, ...] = ()
    exact: bool = False


@dataclass(frozen=True)
class SuccessSpec:
    must_restore: tuple[MetricRequirement, ...]
    must_not_break: tuple[MetricRequirement, ...]
    required_actions: ActionRequirement
    forbidden_actions: tuple[ActionMatcher, ...]
    max_steps: int


@dataclass(frozen=True)
class Scenario:
    id: str
    seed: int
    initial_state: NetworkState
    sla: dict[str, Threshold]
    success: SuccessSpec
    max_steps: int
