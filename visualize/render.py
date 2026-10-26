"""Render a retargeted G1 motion (kinematic reference, no physics) to MP4.

    python visualize/render.py "task.parquet='<.../robot_name=g1/data.parquet>'" \
        viewer.output_path=<out.mp4> viewer.num_frames=<frames>

This is what FlashCHORD trains the policy to track. It reuses FlashCHORD's own replay app
(flash_chord/scripts/debug/replay.py, sonic_replay config) and only starts it in its
"Show kinematic reference" mode instead of waiting for a click in the Viser GUI.
"""

import sys
from pathlib import Path

FLASH_CHORD = Path(__file__).resolve().parents[1] / "train" / "flash_chord"
sys.path.insert(0, str(FLASH_CHORD / "src"))
sys.path.insert(0, str(FLASH_CHORD / "scripts" / "debug"))

import hydra
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed
from flash_chord.visualization.viewer import ViewerConfig
from replay import Replay


class KinematicReplay(Replay):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self._playing_mode = "reference"


@hydra.main(version_base="1.3", config_path="../train/flash_chord/src/flash_chord/configs", config_name="sonic_replay")
def main(cfg: DictConfig) -> None:
    viewer = instantiate_typed(cfg.viewer, ViewerConfig)
    KinematicReplay.launch_configured(cfg, viewer, device=cfg.device)


if __name__ == "__main__":
    main()
