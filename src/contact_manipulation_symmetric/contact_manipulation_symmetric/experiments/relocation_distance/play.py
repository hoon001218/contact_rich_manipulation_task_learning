"""Play learning 1 with the relocation-distance experiment configuration."""

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from contact_manipulation_symmetric.experiments._play import run_experiment_play

    run_experiment_play("Isaac-Contact-Manipulation-Symmetric-UR5e-RelocationDistance-v0")
