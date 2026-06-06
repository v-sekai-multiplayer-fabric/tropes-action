# tropes-action

A static, no-LLM GitHub Action that fails the build when Markdown / Quarto
contains common AI-writing tells, following the [tropes.fyi](https://tropes.fyi/)
house style. Pure `bash` + `grep` — no model calls, no network, fast.

It scans tracked files with `git ls-files` and reports each hit as
`tropes(<group>): <file>:<line>: <text>`. There are five rule groups. The
`parallelism` group catches negative parallelism (the `not X, but Y` shape and
its em-dash reframe). The `bold-list` group catches a bullet or number that opens
with a bolded label then a colon, period, or em-dash. The `serves-as` group
catches the "serves as a" dodge. The `fragment` group catches rhetorical
fragments such as the "Not X. Not Y." cascade and the "The X? A Y." question.
The `cliche` group catches overused AI vocabulary (the kind collected on
tropes.fyi). Each pattern requires enough context that ordinary prose does not
trip it.

> This README documents the patterns by example, so it deliberately contains
> phrasings the checker flags. The repo therefore scans only `tests/`, not its
> own docs; consumers point the action at their content.

## Usage

```yaml
name: Tropes
on: [pull_request]
jobs:
  tropes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: v-sekai-multiplayer-fabric/tropes-action@main
        # optional: restrict what gets scanned (git pathspecs)
        # with:
        #   paths: "docs/**/*.md *.qmd"
```

> Pre-release: track `@main` (or pin a commit SHA) for now. A `vN` tag is cut
> only after the action clears the dev, beta, and rc (release candidate) stages.

| Input          | Default      | Description                                                            |
| -------------- | ------------ | ---------------------------------------------------------------------- |
| `paths`        | `*.md *.qmd` | Space-separated git pathspecs to scan.                                 |
| `ml`           | `false`      | Also run the advisory ONNX classifier (lazy-fetched, CPU, non-blocking). |
| `ml-threshold` | `0.6`        | `p(AI-tell)` threshold for the advisory ML pass.                       |

The regex rules are the deterministic pass/fail gate. With `ml: true` the action
also lazy-downloads a small AutoGluon-trained, ONNX-exported classifier and scores
prose on the CPU, printing warnings (with the phrases that drove each flag) without
failing the build. See [`ml/`](ml/README.md).

## Run locally / as a pre-commit hook

```bash
./check_tropes.sh                 # all Markdown + Quarto
./check_tropes.sh 'decisions/**/*.md'
```

As a [pre-commit](https://pre-commit.com/) / prek local hook:

```yaml
- repo: local
  hooks:
    - id: tropes
      name: tropes (no AI tells)
      entry: bash check_tropes.sh
      language: system
      types: [markdown]
      pass_filenames: false
```

## Why static

A regex check is deterministic, reviewable, and runs in well under a second with
no API key. It will not catch everything an LLM reviewer would, but it reliably
stops the highest-frequency tells from accreting in committed docs.
