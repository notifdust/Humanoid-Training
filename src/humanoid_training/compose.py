from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Standing G1 wrist workspace is roughly x∈[0.15, 0.45], z∈[0.70, 0.95].
# A floor-height counter at x=0.50 is not reachable with arm/waist only.
KITCHEN_COUNTER = {
    "table_pos": (0.28, 0.0, 0.66),
    "table_size": (0.12, 0.12, 0.02),
    "table_rgba": (0.45, 0.32, 0.18, 1.0),
    "camera_pos": (1.10, -0.90, 1.05),
    "camera_target": (0.12, -0.04, 0.78),
}

STAND_CAMERA = {
    "camera_pos": (1.35, -1.00, 1.05),
    "camera_target": (0.0, 0.0, 0.75),
}

PRIMITIVES: dict[str, dict[str, Any]] = {
    "ycb-mustard": {
        "geom": "cylinder",
        "size": (0.035, 0.09, 0.0),
        "rgba": (0.85, 0.70, 0.12, 1.0),
        "mass": 0.4,
        "half_height": 0.09,
    },
    "bowl-white": {
        "geom": "cylinder",
        "size": (0.09, 0.035, 0.0),
        "rgba": (0.93, 0.93, 0.90, 1.0),
        "mass": 0.25,
        "half_height": 0.035,
    },
}

_FALLBACK = {
    "geom": "box",
    "size": (0.04, 0.04, 0.04),
    "rgba": (0.55, 0.70, 0.85, 1.0),
    "mass": 0.2,
    "half_height": 0.04,
}

_GEOM_ENUM = {
    "box": 6,  # mjGEOM_BOX
    "sphere": 2,
    "cylinder": 5,
    "capsule": 3,
}


def scene_wants_compose(spec: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    cfg = cfg or {}
    if cfg.get("honors_scene"):
        return True
    scene = spec.get("scene") or {}
    return bool(scene.get("objects") or scene.get("template"))


def table_layout(scene: dict[str, Any] | None) -> dict[str, Any]:
    scene = scene or {}
    layout = dict(KITCHEN_COUNTER)
    if scene.get("table_pos"):
        layout["table_pos"] = tuple(scene["table_pos"])
    if scene.get("table_size"):
        layout["table_size"] = tuple(scene["table_size"])
    return layout


def object_world_pos(obj: dict[str, Any], layout: dict[str, Any]) -> tuple[float, float, float]:
    prim = primitive_for(obj)
    tx, ty, tz = layout["table_pos"]
    sx, sy, sz = layout["table_size"]
    x = float(obj.get("x") or 0.0)
    y = float(obj.get("y") or 0.0)
    z = tz + sz + float(prim["half_height"]) + 0.002
    return (tx + x, ty + y, z)


def primitive_for(obj: dict[str, Any]) -> dict[str, Any]:
    asset = str(obj.get("asset") or obj.get("id") or "")
    return dict(PRIMITIVES.get(asset) or _FALLBACK)


def camera_xyaxes(pos: Any, target: Any) -> list[float]:
    pos_a = np.asarray(pos, dtype=np.float64)
    target_a = np.asarray(target, dtype=np.float64)
    fwd = target_a - pos_a
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-8:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    fwd = fwd / norm
    z_axis = -fwd
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return [float(v) for v in (*x_axis, *y_axis)]


def compose_mjcf(
    mjcf_path: Path,
    scene: dict[str, Any] | None,
    dest_xml: Path | None = None,
    movable: list[str] | None = None,
) -> tuple[Any, str]:
    """Inject a table, catalog objects, and an eval camera into an MJCF via MjSpec."""
    import mujoco

    mjcf_path = Path(mjcf_path)
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    layout = table_layout(scene)
    world = spec.worldbody
    movable_ids = {str(name) for name in (movable or [])}

    table = world.add_body(name="ht_table", pos=list(layout["table_pos"]))
    table.add_geom(
        name="ht_table_geom",
        type=int(getattr(mujoco.mjtGeom, "mjGEOM_BOX", _GEOM_ENUM["box"])),
        size=list(layout["table_size"]),
        rgba=list(layout["table_rgba"]),
    )

    objects = list((scene or {}).get("objects") or [])
    for obj in objects:
        name = _safe_name(str(obj.get("id") or "object"))
        prim = primitive_for(obj)
        pos = object_world_pos(obj, layout)
        kwargs: dict[str, Any] = {"name": name, "pos": list(pos)}
        if name in movable_ids:
            kwargs["mocap"] = True
        body = world.add_body(**kwargs)
        geom_type = int(
            getattr(
                mujoco.mjtGeom,
                f"mjGEOM_{str(prim['geom']).upper()}",
                _GEOM_ENUM.get(str(prim["geom"]), _GEOM_ENUM["box"]),
            )
        )
        body.add_geom(
            name=f"{name}_geom",
            type=geom_type,
            size=list(prim["size"]),
            rgba=list(prim["rgba"]),
            mass=float(prim["mass"]),
        )

    cam_pos = layout["camera_pos"]
    cam_target = layout["camera_target"]
    world.add_camera(
        name="ht_eval",
        pos=list(cam_pos),
        xyaxes=camera_xyaxes(cam_pos, cam_target),
    )
    world.add_light(pos=[0.4, -0.5, 2.2], dir=[0.1, 0.2, -1.0])

    model = spec.compile()
    xml = spec.to_xml()
    if dest_xml is not None:
        dest_xml = Path(dest_xml)
        dest_xml.parent.mkdir(parents=True, exist_ok=True)
        dest_xml.write_text(xml, encoding="utf-8")
    return model, xml


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return cleaned or "object"
