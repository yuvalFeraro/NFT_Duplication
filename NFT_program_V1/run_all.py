"""
run_all.py
==========
Convenience entry-point that runs all 4 pipeline steps in order.

Usage:
    python run_all.py              # runs steps 1-4
    python run_all.py --steps 2 3  # runs only steps 2 and 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

STEPS = {
    1: SCRIPTS_DIR / "step1_verify_data.py",
    2: SCRIPTS_DIR / "step2_split_data.py",
    3: SCRIPTS_DIR / "step3_extract_hashes.py",
    4: SCRIPTS_DIR / "step4_train_and_test.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run NFT_program_V1 pipeline.")
    parser.add_argument(
        "--steps", nargs="+", type=int, default=list(STEPS.keys()),
        help="Steps to run (default: all)"
    )
    args = parser.parse_args()

    for step_n in sorted(args.steps):
        if step_n not in STEPS:
            print(f"Unknown step: {step_n}. Valid steps: {list(STEPS.keys())}")
            sys.exit(1)
        script = STEPS[step_n]
        print(f"\n{'='*60}")
        print(f"Running Step {step_n}: {script.name}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            print(f"\n[FAIL]  Step {step_n} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print("\n[PASS]  All requested steps completed successfully.")


if __name__ == "__main__":
    main()
