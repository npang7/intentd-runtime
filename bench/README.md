# Operational tooling

These commands run from the repository root. Generated outputs belong under the
ignored `results/` or `runs/` directories.

## Local recovery checks

```sh
python -m examples.recovery_demo
python -m bench.chaos --n 150
```

Both use deterministic stage responses and real SQLite journals. The first
prints a short walkthrough; the second checks interrupted runs across scenarios
and transaction boundaries.

## Optional provider runs

The CLI also supports the Anthropic adapter. Supply `ANTHROPIC_API_KEY` through
the process environment. Provider calls incur charges; keys are not command-line
arguments and are not recorded in journals.

Inspect the available options before running a provider experiment:

```sh
python -m bench.run_eval --help
python -m bench.measure_prefix --help
```

Choose a fresh output file for each experiment. The runner treats completed
scenario records in that file as progress, including runs whose final grade
failed. Prefix measurement associates cache eligibility with the checkout SHA.

## Usage analysis

```sh
python -m bench.analyze --show-pricing
```

The analyzer reports its configured rates and date. To summarize a result file
you have produced, pass `--input results/experiment.jsonl` and
`--output results/summary.md` to `python -m bench.analyze`. It uses the recorded
input, output, cache-write, and cache-read token fields, and keeps transport retries
separate from schema repairs. See the [design notes](../docs/design.md) for the
accounting and delivery boundary.
