# `pvt/` — NeqSim → OLGA `.tab` table generation

Generates the two-phase `PVTTABLE` keyword table that OLGA reads (`FILES PVTFILE=./<name>.tab`).
One command produces a table for a detailed C1–C20 composition over a rectangular
pressure × temperature grid.

This part is solver-agnostic: the same table feeds the steady solver in `steady/`
(`steady/pvt_provider.py`, `steady/pvtsim_tab_interpolator.py`) and the transient cases run
through `transient/`.

## Requirements

```bash
pip install -r pvt/requirements-neqsim.txt      # neqsim==3.16.0, numpy
```

Plus a 64-bit JDK 17+ (`JAVA_HOME`, or `java` on `PATH` — JPype starts a JVM).

> **Pin the version.** `neqsim==3.16.0` is required for byte-identical reproduction. Re-running
> the same grid on `neqsim==3.18.0` changes 27 of 525 points in the last digit (max relative
> difference 3.6e-7) — grid, header, column names and label are unchanged. The JDK version is
> not the cause. For trend comparison only, any recent NeqSim will do.

## Usage

Run from this directory, so `neqsim_tab` is importable:

```bash
cd pvt
python -m neqsim_tab \
    --composition-csv compositions/LEAN_NGC_C20_detailed_C1_C20_renormalized_molpct.csv \
    --composition-column LEAN_NGC_C20_C1_C20_mol_pct \
    --output-tab out/LEAN_NGC_C20_20_150bara_10_60C_OLGA.tab \
    --pressure-min-bara 20 --pressure-max-bara 150 --pressure-count 25 \
    --temperature-min-c 10 --temperature-max-c 60 --temperature-count 21 \
    --fluid-label FLUID-A
```

A 25 × 21 grid (525 points) takes roughly 15–25 s including JVM start-up.

### Outputs

Three files, same directory and stem:

| File | Contents |
| --- | --- |
| `<stem>.tab` | the OLGA two-phase `PVTTABLE` keyword table, `PHASE = TWO` |
| `<stem>_flash_audit.csv` | flash audit at three corners (low / mid / high), SI units, with a local Peng–Robinson cross-check |
| `<stem>_manifest.json` | component mapping, JDK path, unit contract, table structure check |

### Arguments

| Flag | Notes |
| --- | --- |
| `--composition-csv` | composition file, see format below |
| `--composition-column` | **must equal the header name**, i.e. `<fluid>_C1_C20_mol_pct` |
| `--output-tab` | output path |
| `--pressure-min-bara` / `--pressure-max-bara` / `--pressure-count` | pressure axis, bara, uniforms |
| `--temperature-min-c` / `--temperature-max-c` / `--temperature-count` | temperature axis, °C |
| `--fluid-label` | **must match the `FLUID` keyword of the BRANCH in the genkey**, otherwise OLGA reports that the fluid cannot be found |

## Composition input format

CSV with header `component,formula,<fluid>_C1_C20_mol_pct[,phase_role]`:

```csv
component,formula,LEAN_NGC_C20_C1_C20_mol_pct,phase_role
Nitrogen,N2,1,non_hydrocarbon
Carbon dioxide,CO2,2,non_hydrocarbon
Methane,C1,74.97,light_gas
...
```

- Components limited to N2, CO2 and detailed C1–C20 — no plus fraction, no water.
- The selected column must sum to **exactly 100 mol%**. The generator raises on anything else;
  it does not silently renormalise. The files in `compositions/` are already renormalised.
- Only a gas phase and a hydrocarbon liquid phase are generated. There is no aqueous phase and
  no water-phase compensation.

## Unit contract

The generator handles three unit systems, and mixing them up is the main failure mode:

| Layer | Pressure | Temperature |
| --- | --- | --- |
| This project's internal API | Pa | K |
| NeqSim API | bara | K |
| OLGA `.tab` | Pa | °C |

Composition is mol/mol throughout. The emitted table is in OLGA's conventions; the audit CSV
reports SI.

## Bundled compositions

`compositions/` holds four generic gas-condensate fluid definitions used as examples. The
pressure ceiling differs per fluid; the temperature axis is 10–60 °C for all of them.

| Fluid | P range (bara) | Column name |
| --- | --- | --- |
| `HGC_A` | 20–120 | `HGC_A_C1_C20_mol_pct` |
| `HGC_B` | 20–100 | `HGC_B_C1_C20_mol_pct` |
| `HGC_C` | 20–90 | `HGC_C_C1_C20_mol_pct` |
| `LEAN_NGC_C20` | 20–150 | `LEAN_NGC_C20_C1_C20_mol_pct` |

## Notes

- The source files in `neqsim_tab/` are kept verbatim from the originating project; only
  `__init__.py` and `__main__.py` are added so the package can be invoked with `python -m`.
- Tables are rectangular and dense. Grid corners are the flashing extremes: too narrow a
  temperature range and OLGA will stop with `TEMPERATURE BELOW TABLE VALUES` (return code 68)
  once Joule–Thomson cooling walks out of the table.
