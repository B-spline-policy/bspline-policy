"""
Usage:
cd bspline_policy
python train.py --config-name=clean_bspline_policy_unet_bspline
"""

import pathlib
import sys

BSPLINE_POLICY_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(BSPLINE_POLICY_DIR.joinpath("bspline_policy", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
