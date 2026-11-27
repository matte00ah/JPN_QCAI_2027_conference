"""Block 'detection' of the experiment: runs its steps in order (see qc_pipeline.py and --list).

    python Experiments/qcai2027/run_detection.py [--device cuda|cpu] [--list] [--from N] [--to N] [--dry-run]
"""
import sys

from qc_pipeline import run_block

if __name__ == "__main__":
    sys.exit(run_block("detection"))
