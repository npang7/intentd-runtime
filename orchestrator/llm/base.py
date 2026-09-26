"""Provider-neutral model interface and call records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class TransportRetry:
    error_category: str
    status_code: int | None
    transport_attempt: int
    wait_s: float
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class LLMCall:
    stage: str
    agent: str
    attempt: int
    latency_s: float
    raw_response: str
    parsed_ok: bool
    model: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    transport_retries: tuple[TransportRetry, ...] = ()


class LLMClient(ABC):
    @property
    @abstractmethod
    def model(self) -> str:
        """Return a stable identifier suitable for persisted run records."""

    @property
    @abstractmethod
    def calls(self) -> Sequence[LLMCall]:
        """Return completed calls in invocation order."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        schema_name: str,
        *,
        stage: str,
        agent: str,
        attempt: int,
    ) -> str:
        """Return provider text without replacing it with a parsed model."""
