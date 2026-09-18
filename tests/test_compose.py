from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_training.compose import compose_mjcf, object_world_pos, table_layout


def test_compose_adds_table_objects_and_camera() -> None:
    mjcf = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    scene = {
        "template": "kitchen-counter-v1",
        "objects": [
            {"id": "mustard", "asset": "ycb-mustard", "x": -0.18, "y": 0.04},
            {"id": "bowl", "asset": "bowl-white", "x": 0.16, "y": -0.02},
        ],
    }
    model, xml = compose_mjcf(mjcf, scene)
    assert "ht_table" in xml
    assert "mustard" in xml
    assert "ht_eval" in xml
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    layout = table_layout(scene)
    mustard = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mustard")
    want = object_world_pos(scene["objects"][0], layout)
    got = data.xpos[mustard]
    assert np.hypot(got[0] - want[0], got[1] - want[1]) < 0.01
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ht_eval")
    assert cam >= 0
