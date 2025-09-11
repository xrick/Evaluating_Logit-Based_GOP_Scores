# Repository Guidelines

## Project Structure & Module Organization
- `official/`: Canonical code for the paper; subfolders `MPC/`, `SO/`, and evaluation notebooks in `MPC_eval/`, `SO_eval/`. Uses `requirements.txt`, writes to `official/cache_dir/` and `official/output/`.
- `myImplementation/`: Experimental methods and prototypes (e.g., `speechocean_experiments.py`, `Src1/`, `Src2/`).
- `docs/`: Papers, how-to guides, and analyses (see `docs/how_to_run.md`).
- Root utilities: `convert2hf_*.py`, `GenHF*.sh` for dataset prep and conversions.

## Build, Test, and Development Commands
- Create env and install deps (official pipeline):
  - `cd official && pip install -r requirements.txt`
- Run MPC quantification:
  - `cd official/MPC && python mpc_ctc_segment_quantification.py`
- Run SpeechOcean quantification:
  - `cd official/SO && python speechocean_quantification.py`
- Evaluate (notebooks):
  - `cd official/MPC_eval && jupyter notebook mpc_evaluate.ipynb`
  - `cd official/SO_eval && jupyter notebook so_evaluation.ipynb`
- Experimental script example:
  - `python myImplementation/speechocean_experiments.py`

## Coding Style & Naming Conventions
- Python 3.8+; follow PEP 8.
- Indentation: 4 spaces; max line length 100.
- Names: modules `lower_snake_case.py`, functions/vars `lower_snake_case`, classes `CamelCase`.
- Prefer type hints and docstrings; keep side effects behind `if __name__ == "__main__":`.
- Large data paths configurable at top-of-file constants; avoid hard‑coding user-specific paths in commits.

## Testing Guidelines
- No formal unit test suite. Validate changes by:
  - Running the quantification scripts on a small subset and inspecting CSV outputs.
  - Re-running the evaluation notebooks to confirm metrics/plots.
- Add lightweight checks where feasible (e.g., shape/column assertions before writing CSVs).
- Keep runs deterministic (set `random_state`/seeds when applicable).

## Commit & Pull Request Guidelines
- Use Conventional Commits style for clarity (e.g., `feat: add SO quantification flag`, `fix: handle empty segment edge case`).
- Keep commits focused; include rationale in the body when behavior changes.
- PRs must include:
  - Purpose and scope; linked issue (if any).
  - Repro steps and example commands (paths, env, dataset subset).
  - Notes on outputs written (CSV names/locations) and any plotting changes.
  - Screenshots or metrics diffs when modifying evaluations.

## Security & Configuration Tips
- Do not commit datasets, large artifacts, or secrets. Use `.gitignore`d `cache_dir/` and `output/`.
- If paths differ locally, document overrides in the PR and reference `docs/how_to_run.md`.
