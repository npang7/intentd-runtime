# Runtime design

intentd separates service decisions from durable execution. The coordinator moves
a scenario through telemetry, policy, validation, and execution. Contracts make
each stage's output explicit; the journal records both progress and effects.

## One transaction for one logical effect

The SQLite journal has three tables:

| Table | Responsibility |
|---|---|
| `runs` | Scenario, model identifier, current state, and run status |
| `events` | Ordered stage, model-call, and tool-call records per run |
| `effects` | Idempotency keys, action arguments, and committed results |

An action updates `runs.state_json`, inserts its `effects` row, and appends a
`tool_called` event inside one `BEGIN IMMEDIATE` transaction. A failed event insert
rolls back the state and effect as well. SQLite uses WAL mode and `synchronous=FULL`.

The idempotency key is a SHA-256 digest of the run ID, stage, logical step, tool,
and canonical arguments. Repeating that logical step returns its stored result.
The `effects` primary key also protects against a duplicate insert. Identical
arguments at different steps remain different logical actions.

This atomic boundary works because simulated state and the journal share a
database. A physical device would require its own idempotency or reconciliation
mechanism; a database transaction alone does not make an external side effect atomic.

## Recovery follows the execution path

Recovery reads the persisted state, completed stage outputs, and actions ordered
by logical step. It validates saved outputs with the same contracts used during a
normal run, then calls the same orchestration graph with that progress.

- Completed telemetry, policy, and validation stages are reused.
- If execution stopped partway through, the journal's effect lookup skips committed
  steps while the remaining actions execute.
- A completed run returns its saved stage outputs and a grade of its stored state;
  it does not repeat model calls or actions.

The fault-injection harness exits real child processes at two transaction
boundaries: before commit and after commit but before the caller continues. It
compares recovery with a clean reference using the complete state hash, effect
count, and tool-event count. The runnable recovery example selects the latter
boundary for the first configuration action.

## Concurrency with a single persistence owner

Cell telemetry tasks share a semaphore limit and use independent model clients.
The coordinator awaits the tasks, sorts their records by cell ID, and merges
findings in that order. Worker tasks do not touch SQLite.

This keeps concurrency in the slow service calls rather than in storage writes.
It also makes journal order independent of task completion order. Serial telemetry
uses the same inputs and merge path, which allows tests to compare both schedules.

Call records are buffered until the telemetry stage is collected. A crash during
fan-out can therefore repeat calls from that unfinished stage. Deterministic
fixtures reproduce those calls; a live service can produce a different response
and additional usage. Tool-effect idempotency does not imply exactly-once model
delivery.

## Contracts and distinct retry budgets

Stage outputs use strict Pydantic models: undeclared fields and coercion are
rejected, and configuration parameters have explicit bounds. The provider adapter
translates the output schema to the provider's supported subset while preserving
the local validation checks.

Schema repair sends structured feedback about malformed JSON or invalid fields
and has a finite attempt budget. Transport retries handle connection failures,
timeouts, and selected HTTP statuses separately. Each category has its own journal
records and counters, so valid-output retries are not conflated with delivery errors.

The journal retains raw token usage, cache token fields, request IDs, and latency.
Cost analysis consumes those fields rather than estimating them from text length.
Responses lost in transit may leave provider work unobserved; local accounting
reports the usage actually received.

## Reproducible application state

The simulator defines 18 named scenarios across eight network-operation families.
Configuration actions transform immutable state, and deterministic metric functions
derive the resulting telemetry. A code-based grader checks metric requirements,
action rules, and action-count budgets.

Standard and trap sequences exercise both successful outcomes and plausible
incorrect actions. The deterministic model adapter supplies these sequences for
runtime verification, keeping service variability separate from persistence and
recovery tests. See the [testing guide](../tests/README.md) and
[runnable examples](../README.md#run-a-deterministic-demo).
