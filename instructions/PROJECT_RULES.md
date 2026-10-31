# NFT_program_V2 — Project Rules & AI Guidelines

These instructions must be strictly followed before writing code, running commands, or creating files in this project:

## 1. No Loose Scripts in the Root Folder
- Never create ad-hoc `.py` scripts in the root directory.
- All functional code must be organized inside dedicated subdirectories:
  - `src/` for core pipeline code (BK-Tree, XGBoost, feature extraction).
  - `experiments/` for benchmark runs and evaluations.
  - `scripts/temp/` for any temporary data wrangling.

## 2. No Scratch Scripts for Simple Checks
- Do NOT generate throwaway scripts (like `check_excel.py` or `inspect_data.py`).
- For quick data checks, run a terminal one-liner instead:
  `python -c "import pandas as pd; df=pd.read_excel('...'); print(df.columns)"`

## 3. Update Existing Code — Do Not Create Versioned Duplicates
- Avoid creating files with dates or duplicate names (e.g., `model_2409.py`, `script_v2_new.py`).
- Update existing files directly or use configurable command-line arguments.

## 4. Temporary File Cleanup
- If a multi-step helper script is strictly needed, place it in `scripts/temp/` and remove it once the task is complete.

## 5. Confirmation Before Adding Files
- Before creating any new file, summarize why it is necessary and confirm with the user.

## 6. Architecture & Trade-Off Evaluations
- When analyzing or recommending the best options, always evaluate:
  - **Scalability**: Will it scale efficiently to massive datasets (time & memory complexity, e.g., $O(\log N)$ retrieval vs $O(N)$)?
  - **Recall & FPR**: Will recall improve while strictly maintaining or reducing False Positive Rate (FPR)?

## 7. Reference Prior Work in `NFT_program`
- Consult the sibling folder `NFT_program` (`../NFT_program`) under the parent directory `NFT_Duplication_Prj`.
- Many utilities, hash routines, and data processing patterns can be adapted from or cross-referenced with existing implementations there instead of reinventing them.
