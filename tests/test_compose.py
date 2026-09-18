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


def test_kitchen_counter_sits_in_standing_g1_workspace() -> None:
    from humanoid_training.compose import KITCHEN_COUNTER, object_world_pos, table_layout

    layout = table_layout({})
    assert layout["table_pos"] == KITCHEN_COUNTER["table_pos"]
    mustard = object_world_pos(
        {"id": "mustard", "asset": "ycb-mustard", "x": -0.10, "y": -0.05},
        layout,
    )
    # Standing G1 wrist workspace is roughly x∈[0.15, 0.45], z∈[0.70, 0.95].
    assert 0.15 <= mustard[0] <= 0.40
    assert 0.70 <= mustard[2] <= 0.90


def test_compose_mocap_mustard() -> None:
    mjcf = Path(__file__).resolve().parent / "fixtures" / "mini_humanoid.xml"
    scene = {
        "template": "kitchen-counter-v1",
        "objects": [
            {"id": "mustard", "asset": "ycb-mustard", "x": -0.18, "y": 0.04},
            {"id": "bowl", "asset": "bowl-white", "x": 0.16, "y": -0.02},
        ],
    }
    model, xml = compose_mjcf(mjcf, scene, movable=["mustard"])
    assert "mocap" in xml.lower()
    mustard = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mustard")
    assert int(model.body_mocapid[mustard]) >= 0
