#!/usr/bin/env python3
"""Export UR5 + Robotiq xacro to MuJoCo XML with relative STL mesh paths.

Pipeline:
1) Expand xacro to URDF.
2) Remove ROS/Gazebo tags MuJoCo does not parse.
3) Convert mesh references to STL files under a relative assets/ directory.
4) Add <mujoco><compiler meshdir="assets" .../></mujoco> to URDF.
5) Compile URDF with MuJoCo and save final MJCF XML.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

DEFAULT_WS = Path("/home/amdt/ur_force_ws")
DEFAULT_XACRO = DEFAULT_WS / "my_ws/urdf/ur5_description/urdf/ur5_gripper.urdf.xacro"
DEFAULT_XML_OUT = DEFAULT_WS / "my_ws/urdf/ur5_gripper.mujoco.xml"
DEFAULT_URDF_OUT = DEFAULT_WS / "my_ws/urdf/ur5_gripper.mujoco.urdf"

# MuJoCo collapses fixed joints during compilation. Preserve these names by
# converting them to zero-range revolute joints in the intermediate URDF.
PRESERVE_FIXED_JOINTS = {
    "robotiq_85_left_finger_joint",
    "robotiq_85_right_finger_joint",
}

CHECK_ROBOTIQ_JOINTS = {
    "robotiq_85_left_finger_joint",
    "robotiq_85_right_finger_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
}

ZERO_RANGE_JOINTS = {
    "robotiq_85_left_finger_joint",
    "robotiq_85_right_finger_joint",
}


def run_xacro(xacro_path: Path, workspace_root: Path) -> str:
    setup_cmds = ["source /opt/ros/humble/setup.bash"]
    ws_setup = workspace_root / "install/setup.bash"
    if ws_setup.exists():
        setup_cmds.append(f"source {ws_setup}")

    cmd = " && ".join(setup_cmds + [f"ros2 run xacro xacro {xacro_path}"])
    proc = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"xacro expansion failed\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout


def remove_tag_blocks(xml_text: str, tag: str) -> str:
    block_pattern = re.compile(rf"<\s*{tag}\b[^>]*>.*?<\s*/\s*{tag}\s*>", re.DOTALL)
    self_pattern = re.compile(rf"<\s*{tag}\b[^>]*/\s*>", re.DOTALL)
    xml_text = block_pattern.sub("", xml_text)
    xml_text = self_pattern.sub("", xml_text)
    return xml_text


def rewrite_package_uris(xml_text: str, package_roots: dict[str, Path]) -> str:
    def repl(match: re.Match[str]) -> str:
        pkg = match.group(1)
        rel = match.group(2)
        root = package_roots.get(pkg)
        if root is None:
            return match.group(0)
        return str((root / rel).resolve())

    return re.sub(r"package://([^/]+)/([^\"'\s<>]+)", repl, xml_text)


def find_stl_candidate(src: Path) -> Path | None:
    if src.suffix.lower() == ".stl" and src.exists():
        return src

    # Prefer collision STL counterpart when source is a visual mesh.
    candidates = [
        src.with_suffix(".stl"),
        Path(str(src).replace("/visual/", "/collision/")).with_suffix(".stl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def unique_name_for(src: Path, used_names: set[str]) -> str:
    base = src.name
    if base not in used_names:
        used_names.add(base)
        return base
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    name = f"{src.stem}_{digest}{src.suffix}"
    used_names.add(name)
    return name


def relativize_meshes_to_assets(xml_text: str, assets_dir: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    used_names: set[str] = set()
    src_to_local: dict[Path, str] = {}

    mesh_re = re.compile(r'<mesh\s+filename="([^"]+)"')

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        src = Path(raw)
        if not src.is_absolute():
            # Keep unknown relative references unchanged.
            warnings.append(f"relative mesh kept as-is: {raw}")
            return match.group(0)

        stl = find_stl_candidate(src)
        if stl is None:
            warnings.append(f"no STL found for mesh: {raw}")
            return match.group(0)

        local = src_to_local.get(stl)
        if local is None:
            local = unique_name_for(stl, used_names)
            src_to_local[stl] = local
            shutil.copy2(stl, assets_dir / local)

        return f'<mesh filename="assets/{local}"'

    return mesh_re.sub(repl, xml_text), warnings


def ensure_mujoco_compiler(xml_text: str) -> str:
    if "<mujoco>" in xml_text:
        return xml_text
    return re.sub(
        r'(<robot\s+name="[^"]+">)',
        r'\1\n  <mujoco>\n    <compiler meshdir="assets" balanceinertia="true" discardvisual="false"/>\n  </mujoco>',
        xml_text,
        count=1,
    )


def preserve_fixed_joint_names(xml_text: str) -> str:
    joint_block_re = re.compile(r"<joint\b[^>]*name=\"([^\"]+)\"[^>]*>.*?</joint>", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        block = match.group(0)
        name = match.group(1)
        if name not in PRESERVE_FIXED_JOINTS:
            return block
        if 'type="fixed"' not in block:
            return block

        # Keep kinematics equivalent to fixed while preventing compiler collapse.
        block = block.replace('type="fixed"', 'type="revolute"', 1)
        if "<axis" not in block:
            block = block.replace("</joint>", '    <axis xyz="0 0 1"/>\n</joint>', 1)
        if "<limit" not in block:
            block = block.replace(
                "</joint>",
                '    <limit lower="0" upper="0" effort="0.1" velocity="0.1"/>\n</joint>',
                1,
            )
        return block

    return joint_block_re.sub(repl, xml_text)


def report_joint_coverage(xml_path: Path) -> None:
    xml_text = xml_path.read_text(encoding="utf-8")
    found = set(re.findall(r'<joint\s+name="([^"]+)"', xml_text))
    missing = sorted(CHECK_ROBOTIQ_JOINTS - found)
    if missing:
        print("[WARN] Missing expected Robotiq joints in output XML:")
        for name in missing:
            print(f"  - {name}")
    else:
        print("[OK] Robotiq finger/tip joints preserved in output XML.")


def extract_urdf_mimics(urdf_text: str) -> list[tuple[str, str, float, float]]:
    """Return URDF mimic constraints as (joint, source, multiplier, offset)."""
    rules: list[tuple[str, str, float, float]] = []
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError as exc:
        print(f"[WARN] Failed to parse URDF for mimic extraction: {exc}")
        return rules

    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        mimic = joint.find("mimic")
        if mimic is None:
            continue

        source = mimic.get("joint")
        if not source:
            continue
        multiplier = float(mimic.get("multiplier", "1.0"))
        offset = float(mimic.get("offset", "0.0"))
        rules.append((name, source, multiplier, offset))

    return rules


def apply_mimic_constraints_to_mjcf(
    xml_path: Path,
    mimic_rules: list[tuple[str, str, float, float]],
    zero_range_joints: set[str],
) -> None:
    """Inject MuJoCo equality joint constraints from URDF mimic relationships."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mj_joints = {
        joint.get("name")
        for joint in root.iter("joint")
        if joint.get("name")
    }

    # Keep intended fixed-like joints pinned after compilation.
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in zero_range_joints:
            joint.set("range", "0 0")

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    existing_pairs = {
        (j.get("joint1"), j.get("joint2"))
        for j in equality.findall("joint")
    }

    injected = 0
    for joint_name, source_name, multiplier, offset in mimic_rules:
        if joint_name not in mj_joints or source_name not in mj_joints:
            continue

        pair = (joint_name, source_name)
        if pair in existing_pairs:
            continue

        ET.SubElement(
            equality,
            "joint",
            {
                "joint1": joint_name,
                "joint2": source_name,
                # q_joint1 = offset + multiplier * q_joint2
                "polycoef": f"{offset:g} {multiplier:g} 0 0 0",
            },
        )
        existing_pairs.add(pair)
        injected += 1

    # Avoid emitting an empty <equality/> when no constraints are present.
    if len(equality.findall("joint")) == 0:
        root.remove(equality)

    tree.write(xml_path, encoding="utf-8", xml_declaration=False)
    if injected:
        print(f"[OK] Injected {injected} mimic equality constraints into XML.")
    else:
        print("[WARN] No mimic equality constraints injected into XML.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export xacro to MuJoCo XML with relative STL assets")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WS)
    parser.add_argument("--xacro", type=Path, default=DEFAULT_XACRO)
    parser.add_argument("--out-xml", type=Path, default=DEFAULT_XML_OUT)
    parser.add_argument("--out-urdf", type=Path, default=DEFAULT_URDF_OUT)
    parser.add_argument("--assets-dir", type=Path, default=None)
    args = parser.parse_args()

    ws = args.workspace.resolve()
    xacro_path = args.xacro.resolve()
    out_xml = args.out_xml.resolve()
    out_urdf = args.out_urdf.resolve()
    assets_dir = (args.assets_dir.resolve() if args.assets_dir else out_xml.parent / "assets")

    if not xacro_path.exists():
        print(f"[ERROR] xacro not found: {xacro_path}")
        return 2

    package_roots = {
        "ur5_description": ws / "my_ws/urdf/ur5_description",
        "robotiq_description": ws / "my_ws/urdf/robotiq_description",
    }

    try:
        xml = run_xacro(xacro_path, ws)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 3

    for tag in ("gazebo", "transmission", "plugin"):
        xml = remove_tag_blocks(xml, tag)

    xml = rewrite_package_uris(xml, package_roots)

    assets_dir.mkdir(parents=True, exist_ok=True)
    # Keep directory clean to avoid stale meshes from prior exports.
    for p in assets_dir.glob("*.stl"):
        p.unlink()

    xml, warnings = relativize_meshes_to_assets(xml, assets_dir)
    mimic_rules = extract_urdf_mimics(xml)
    xml = preserve_fixed_joint_names(xml)
    xml = ensure_mujoco_compiler(xml)

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(xml, encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_path(str(out_urdf))
        mujoco.mj_saveLastXML(str(out_xml), model)
        apply_mimic_constraints_to_mjcf(out_xml, mimic_rules, ZERO_RANGE_JOINTS)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] MuJoCo compile failed: {exc}")
        print(f"[INFO] Debug URDF kept at: {out_urdf}")
        return 4

    print(f"[OK] URDF: {out_urdf}")
    print(f"[OK] XML : {out_xml}")
    print(f"[OK] assets dir: {assets_dir}")
    print(f"[OK] njnt={model.njnt}, nbody={model.nbody}, nq={model.nq}, nv={model.nv}")
    report_joint_coverage(out_xml)
    if warnings:
        print("[WARN] Some meshes were not rewritten to STL:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
