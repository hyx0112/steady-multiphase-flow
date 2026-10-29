# `transient/` — OLGA batch invocation

One file, `olga_batch_flow.py`, standard library only. It takes a directory of ready-made
`.genkey` cases, runs OLGA on each one concurrently, judges completion, and parses the `.tpl`
trend files into a single long CSV.

This covers **the execution side only**. It does not generate or parameterise `.genkey` files —
you bring those.

The file is kept verbatim from the campaign it ran (3,544 cases), so its docstring, inline
comments and `--help` strings are in Chinese. This README is the English reference for the
interface. On a non-UTF-8 console, prefix `python -X utf8` to read the help cleanly.

## Requirements

- Python ≥ 3.10, no third-party packages
- An OLGA installation. The solver is auto-detected from
  `OLGA*/OlgaExecutables/OLGA-*.exe`, or pass `--solver`, or set `OLGA_EXE`.
- A valid licence: the tool reads `SLBSLS_LICENSE_FILE` / `LM_LICENSE_FILE`, or pass `--license`,
  or let it auto-detect.

## Usage

```bash
# 1) preflight: solver / licence / genkey / PVT all reachable?
python transient/olga_batch_flow.py check --genkey-dir cases/ --pvt-dir pvt_tables/

# 2) run (smoke first with --limit 3 --dry-run, then for real)
python transient/olga_batch_flow.py run --genkey-dir cases/ --pvt-dir pvt_tables/ \
    --work runs/ --workers 8
#    interrupted? rerun the same command with --resume
#    want the command lines without executing?  --dry-run

# 3) status
python transient/olga_batch_flow.py status --work runs/

# 4) .tpl trends -> long CSV
python transient/olga_batch_flow.py curves --work runs/ --out curves.csv --every 10
python transient/olga_batch_flow.py curves --tpl one.tpl --out one.csv      # single file
```

## Input contract

- One case = one `<case_id>.genkey` plus the PVT table it references.
- The genkey must contain a line `FILES PVTFILE=./<name>.tab`. The tool copies that table into
  the case working directory, because OLGA only looks for it in the current directory.
- Case list, either way:
  - `--genkey-dir DIR` — recursive scan for `*.genkey`, case id = filename stem;
  - `--cases cases.csv` — columns `case_id,genkey[,pvt][,fluid][,scenario]`.
- Keep the working directory on a pure-ASCII path.

## How "finished" is decided

A case counts as finished only when **all three** hold:

1. `.out` contains `TOTAL EXECUTION TIME:`;
2. `.tpl` exists;
3. `.ppl` exists.

Exit code is recorded but not used as the criterion. A non-zero `rc` with all three artefacts
present is still recorded as finished, with `rc` kept verbatim — attribution is left to you.
Checking only for `.tpl` is not enough: OLGA appends to `.tpl` while running, so a partial file
looks like a finished case. `curves` applies the same gate (override with `--allow-partial`).

## Outputs

| Path | Contents |
| --- | --- |
| `<work>/<case_id>/` | raw artefacts (`.out`, `.tpl`, `.ppl`, `.log`, `.rsw`, …) |
| `<work>/_logs/<case_id>.log` | stdout + stderr |
| `<work>/_status/<case_id>.json` | per-case status, including every attempt's `rc` and elapsed time |
| `<work>/run_summary.csv` | one row per case |
| `curves --out` | long table (see below) |

`curves` columns: `case_id, pipe, branch, scope, nr, x_m, t_s, PT_MPa, TM_C, HOL, UG_m_s, UL_m_s`.
One time series per (pipe, cross-section); `x_m` is the cell centre; `scope` separates
`SECTION` from `BOUNDARY`; `t_s` keeps four decimals. `--every N` thins the time axis but always
keeps the last instant.

With several PIPE segments under one BRANCH, pass `--pipe-segments PIPE-1=100,PIPE-2=50` to set
the cell count per pipe; otherwise the total is split evenly across pipes, which will be wrong.

Output paths ending in `.gz` are gzip-compressed.

## Not in scope

- genkey authoring / parameter sweeps (input is an existing `.genkey`);
- alignment against a reference solver or error metrics — that is downstream.

See [`../pvt/`](../pvt/) for generating the PVT table that the genkey references.
