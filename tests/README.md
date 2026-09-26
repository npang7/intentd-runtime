# Testing

Run checks from the repository root with the development environment active:

```sh
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
```

Tests use temporary SQLite databases and local model fixtures. Provider adapter
tests use controlled SDK responses, so the suite requires no service credentials.

## Focused checks

| Behavior | Command |
|---|---|
| Atomic state, effect, and event writes | `python -m pytest tests/test_transaction_atomicity.py` |
| Logical-step idempotency | `python -m pytest tests/test_idempotency.py` |
| Completed-stage and partial-execution recovery | `python -m pytest tests/test_resume.py` |
| Bounded fan-out, ordering, and scheduling equivalence | `python -m pytest tests/test_fanout_concurrency_limit.py tests/test_fanout_equivalence.py tests/test_fanout_speedup.py` |
| Contracts and repair feedback | `python -m pytest tests/test_contracts.py tests/test_repair_loop.py tests/test_repair_message.py` |
| Provider transport and usage records | `python -m pytest tests/test_anthropic_client.py tests/test_anthropic_journal.py tests/test_result_records.py` |
| Runnable examples and documentation links | `python -m pytest tests/test_examples.py tests/test_documentation.py` |

The transaction tests inject a failing SQLite trigger and verify that every
record rolls back together. Recovery tests interrupt each stage and partially
completed execution, then compare the resumed state and calls with the reference.
Concurrency tests use controlled service delay to exercise scheduling rather than
measure live provider latency.

## Process-termination checks

```sh
python -m bench.chaos --n 150
```

The harness creates clean and interrupted child-process runs, exits at
transaction boundaries, and resumes the interrupted journals. It checks final
state hashes, committed effects, and tool-event counts. CI runs this command in
its own job; the regular test suite includes a smaller smoke test.

The [recovery example](../README.md#watch-crash-recovery) shows one committed-action
interruption and verifies that another resume adds no effects.
