import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BSPLINE_POLICY_DIR = REPO_ROOT / "bspline_policy"
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


if __name__ == "__main__":
    runpy.run_module(
        "bspline_policy.scripts.yam_replay_episodes_bspline",
        run_name="__main__",
    )
else:
    from bspline_policy.scripts.yam_replay_episodes_bspline import *  # noqa: F401,F403
