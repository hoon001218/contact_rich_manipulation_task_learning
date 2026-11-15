"""Standard-library-only delegation to the task's native playback entry point."""

import argparse
from pathlib import Path
import runpy
import sys


def run_experiment_play(task_id: str, argv: list[str] | None = None) -> None:
    """Select one experiment and forward all other CLI/Hydra options unchanged.

    Do not import an EnvCfg here: the shared script must start AppLauncher before
    importing Isaac Lab simulation modules. An explicit matching --task is fine;
    a different task is rejected so this entry point cannot silently play with
    another experiment's configuration.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task", default=task_id)
    selected, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if selected.task != task_id:
        parser.error(f"This playback entry point requires --task {task_id}.")
    script = Path(__file__).resolve().parents[2] / "scripts" / "play.py"
    original_argv = sys.argv
    try:
        sys.argv = [str(script), "--task", task_id, *forwarded]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv
