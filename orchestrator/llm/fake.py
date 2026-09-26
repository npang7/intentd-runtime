"""Offline, deterministic model implementation for tests and benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal, cast

from orchestrator.contracts import response_parses
from orchestrator.llm.base import LLMCall, LLMClient, Message

Mode = Literal["scripted", "cassette", "adversarial", "latency"]

_MODE_ORDER: tuple[Mode, ...] = ("scripted", "cassette", "adversarial", "latency")


def _canonical_prompt(messages: Sequence[Message], schema_name: str) -> str:
    payload = {
        "messages": [
            {"content": message["content"], "role": message["role"]} for message in messages
        ],
        "schema_name": schema_name,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def cassette_key(messages: Sequence[Message], schema_name: str) -> str:
    """Build a stable cassette lookup key from canonical prompt bytes."""

    prompt = _canonical_prompt(messages, schema_name)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class FakeModel(LLMClient):
    """Compose deterministic scripted/cassette responses with test modifiers."""

    def __init__(
        self,
        *,
        modes: Sequence[Mode] = ("scripted",),
        seed: str = "intentd",
        malformed_rate: float = 0.0,
        schema_violation_rate: float = 0.0,
        latency_s: float = 0.0,
        cassette_path: Path | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        requested = tuple(modes)
        unknown = [mode for mode in requested if mode not in _MODE_ORDER]
        if unknown:
            raise ValueError(f"unknown fake modes: {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("fake modes must not be repeated")

        source_count = sum(mode in requested for mode in ("scripted", "cassette"))
        if source_count != 1:
            raise ValueError("choose exactly one response source: scripted or cassette")

        self._validate_rate("malformed_rate", malformed_rate)
        self._validate_rate("schema_violation_rate", schema_violation_rate)
        if malformed_rate + schema_violation_rate > 1.0:
            raise ValueError("malformed_rate and schema_violation_rate must sum to at most one")
        if (malformed_rate or schema_violation_rate) and "adversarial" not in requested:
            raise ValueError("nonzero invalid rates require adversarial mode")
        if not math.isfinite(latency_s) or latency_s < 0.0:
            raise ValueError("latency_s must be finite and nonnegative")
        if latency_s and "latency" not in requested:
            raise ValueError("nonzero latency_s requires latency mode")

        self._modes = tuple(mode for mode in _MODE_ORDER if mode in requested)
        self._seed = seed
        self._malformed_rate = malformed_rate
        self._schema_violation_rate = schema_violation_rate
        self._latency_s = latency_s
        self._sleep = sleep
        self._calls: list[LLMCall] = []
        self._cassette = self._load_cassette(cassette_path) if "cassette" in requested else None

    @staticmethod
    def _validate_rate(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")

    @staticmethod
    def _load_cassette(path: Path | None) -> dict[str, str]:
        if path is None:
            raise ValueError("cassette mode requires cassette_path")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            raise ValueError("cassette must be a JSON object mapping hashes to responses")
        return cast(dict[str, str], data)

    @property
    def model(self) -> str:
        return "fake:" + "+".join(self._modes)

    @property
    def calls(self) -> Sequence[LLMCall]:
        return tuple(self._calls)

    async def complete(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        stage: str,
        agent: str,
        attempt: int,
    ) -> str:
        if "latency" in self._modes:
            # asyncio considers a timer due when when < loop.time() + clock_resolution.
            # This host's monotonic resolution is 15.625ms, so asyncio.sleep calls shorter
            # than about 16ms can return immediately while latency_s uses the sub-microsecond
            # perf_counter, making wall-clock assertions contradictory.
            # Reproduce: python -c "import time; print(time.get_clock_info('monotonic'))"
            # Unit tests therefore assert the requested sleep, not elapsed wall time. Real
            # Concurrency timing tests use durations >=100ms and assert ratios.
            await self._sleep(self._latency_s)

        raw_response = self._source_response(messages, schema_name)
        if "adversarial" in self._modes:
            raw_response = self._apply_adversarial(raw_response, messages, schema_name, attempt)

        self._calls.append(
            LLMCall(
                stage=stage,
                agent=agent,
                attempt=attempt,
                latency_s=self._latency_s,
                raw_response=raw_response,
                parsed_ok=response_parses(schema_name, raw_response),
                model=self.model,
            )
        )
        return raw_response

    def _source_response(self, messages: Sequence[Message], schema_name: str) -> str:
        if "cassette" in self._modes:
            if self._cassette is None:
                raise RuntimeError("cassette mode has no loaded cassette data")
            key = cassette_key(messages, schema_name)
            try:
                return self._cassette[key]
            except KeyError as exc:
                raise KeyError(f"cassette response not found for {key}") from exc
        return self._scripted_response(messages, schema_name)

    @staticmethod
    def _scripted_response(messages: Sequence[Message], schema_name: str) -> str:
        context = "\n".join(message["content"] for message in messages).strip()
        reason = context or "No actionable telemetry was supplied."
        if schema_name == "TelemetryReport":
            payload: object = {"findings": [], "summary": reason}
        elif schema_name == "PolicyPlan":
            payload = {
                "actions": [{"reason": reason, "tool": "no_op"}],
                "expected_effect": "Network state remains unchanged.",
                "rationale": reason,
            }
        elif schema_name == "ValidationVerdict":
            payload = {"approved": True, "revised_actions": [], "violations": []}
        elif schema_name == "ExecutionResult":
            payload = {
                "applied": [{"reason": reason, "tool": "no_op"}],
                "post_metrics": {},
                "skipped": [],
            }
        else:
            raise ValueError(f"unknown schema: {schema_name}")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _apply_adversarial(
        self,
        valid_response: str,
        messages: Sequence[Message],
        schema_name: str,
        attempt: int,
    ) -> str:
        # Output decisions must remain identical across processes. Never use built-in hash(),
        # dict/set iteration order, time, uuid4, or unseeded randomness here or in scripted mode.
        # Draw material may contain only logical-call identity. Instance state and call history
        # would be reset by a recovery run and could make the resumed response diverge.
        material = "\x00".join((self._seed, _canonical_prompt(messages, schema_name), str(attempt)))
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if draw < self._malformed_rate:
            return "{"
        if draw < self._malformed_rate + self._schema_violation_rate:
            return self._schema_violation(valid_response, digest[8])
        return valid_response

    @staticmethod
    def _schema_violation(valid_response: str, selector: int) -> str:
        payload = json.loads(valid_response)
        if selector % 3 == 0 and isinstance(payload, dict) and "actions" in payload:
            payload["actions"] = [
                {"cell_id": "cell-1", "tool": "set_tx_power", "tx_power_dbm": 47.0}
            ]
        elif selector % 3 == 1 and isinstance(payload, dict) and "actions" in payload:
            payload["actions"] = [{"cell_id": "cell-1", "tool": "unknown_tool"}]
        elif isinstance(payload, dict):
            payload["unexpected"] = True
        else:
            payload = {"unexpected": True}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
