#!/usr/bin/env python3
"""根据辨识参数重建 UR5e 的 MuJoCo XML 文件。

功能说明：
1. 读取 Data/identified_params_ur5e.csv 中的动力学辨识结果。
2. 复制 universal_robots_ur5e/ur5e.xml，生成新的
   universal_robots_ur5e/ur5e_real.xml。
3. 将 CSV 中 Joint1-Joint6 的质量、质心、惯量参数写入 XML 中对应的
   6 个 UR5e 运动 body。
4. 将 CSV 中的摩擦参数写入 XML 中对应的 6 个 joint。

重要约定：
- 本脚本只修改 <body> 内部的 <inertial> 参数，以及 <joint> 的
  frictionloss/damping 参数。
- 不修改任何 <body pos="...">、<body quat="..."> 或 joint 的几何位置，
  因此机器人连杆长度、关节安装位置、运动学结构不会改变。
- CSV 中 mx/my/mz 是一次矩，即 mass * center_of_mass。MuJoCo 的
  <inertial pos="..."> 需要的是质心相对当前 body 坐标系的位置，所以
  需要用 mx/m、my/m、mz/m 换算。
- CSV 中 Ixx/Ixy/... 按关于 body 坐标原点的惯量处理。脚本使用平行轴
  定理换算到质心坐标系，再写入 MuJoCo 的 fullinertia。
- Fric1 是静摩擦力/库伦摩擦力矩，写入 joint frictionloss。
- Fric2 是阻尼系数，写入 joint damping。
- Fric3 当前不写入 XML。
- 非对角乘积惯量 ixy/ixz/iyz 如果非常小，会视为数值噪声写成 0。
- 对角主惯量 ixx/iyy/izz 按平行轴定理正常换算后直接写入，不做模板值
  回退。这样输出会忠实反映辨识结果；如果辨识结果不满足 MuJoCo 的惯量
  约束，需要后续单独处理辨识参数或模型约束。
"""

from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# 当前脚本位于 tools/rebuild_xml.py，因此 parents[1] 是 joint_test 根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 辨识参数 CSV 文件。
CSV_PATH = PROJECT_ROOT / "Data" / "identified_params_ur5e.csv"

# 原始 MuJoCo XML 模板。脚本不会直接修改该文件。
SOURCE_XML = PROJECT_ROOT / "universal_robots_ur5e" / "ur5e.xml"

# 写入辨识参数后的输出 XML。
OUTPUT_XML = PROJECT_ROOT / "universal_robots_ur5e" / "ur5e_real.xml"

# 非对角乘积惯量清零阈值，只用于 ixy/ixz/iyz。
OFF_DIAGONAL_INERTIA_ZERO_EPS = 1e-6

# CSV 的列名。Joint1-Joint6 分别对应 UR5e 的 6 个运动关节和运动连杆。
JOINT_COLUMNS = [f"Joint{i}" for i in range(1, 7)]

# MuJoCo XML 中需要更新惯性参数的 body 名称。
# Joint1 -> shoulder_link
# Joint2 -> upper_arm_link
# Joint3 -> forearm_link
# Joint4 -> wrist_1_link
# Joint5 -> wrist_2_link
# Joint6 -> wrist_3_link
BODY_NAMES = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]

# MuJoCo XML 中需要更新摩擦和阻尼参数的 joint 名称。
MJCF_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# CSV 中需要读取的行。
PARAM_ROWS = {
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
    "mx",
    "my",
    "mz",
    "m",
    "Fric1",
    "Fric2",
    "Fric3",
}


def fmt(value: float) -> str:
    """将浮点数格式化为适合 XML 属性的字符串。"""
    return f"{value:.15g}"


def parse_vector(text: str, expected_len: int) -> list[float]:
    """读取 XML 属性中的空格分隔向量。"""
    values = [float(item) for item in text.split()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}: {text}")
    return values


def zero_small_off_diagonal_inertia(value: float) -> float:
    """将过小的非对角乘积惯量清零。"""
    if abs(value) < OFF_DIAGONAL_INERTIA_ZERO_EPS:
        return 0.0
    return value


def load_identified_params(csv_path: Path) -> dict[str, dict[str, float]]:
    """读取 CSV，并整理成按 Joint1-Joint6 访问的参数字典。"""
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        params = {joint: {} for joint in JOINT_COLUMNS}

        for row in reader:
            row_name = row["Row"].strip()
            if row_name not in PARAM_ROWS:
                continue
            for joint in JOINT_COLUMNS:
                params[joint][row_name] = float(row[joint])

    missing = {
        joint: sorted(PARAM_ROWS - values.keys())
        for joint, values in params.items()
        if PARAM_ROWS - values.keys()
    }
    if missing:
        raise ValueError(f"Missing CSV parameters: {missing}")

    return params


def find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    """在整棵 MuJoCo XML 中查找指定 name 的元素。"""
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"Could not find <{tag} name=\"{name}\"> in XML")


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    """确保 parent 下存在指定 tag 的直接子节点。"""
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def central_inertia(params: dict[str, float]) -> dict[str, float]:
    """将关于 body 原点的惯量换算为关于质心的惯量。

    质心坐标：
        x = mx / m, y = my / m, z = mz / m

    平行轴定理：
        I_com_xx = I_origin_xx - m * (y^2 + z^2)
        I_com_yy = I_origin_yy - m * (x^2 + z^2)
        I_com_zz = I_origin_zz - m * (x^2 + y^2)

    乘积惯量按当前辨识参数符号约定换算：
        I_com_xy = I_origin_xy + m * x * y
        I_com_xz = I_origin_xz + m * x * z
        I_com_yz = I_origin_yz + m * y * z
    """
    mass = params["m"]
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")

    x = params["mx"] / mass
    y = params["my"] / mass
    z = params["mz"] / mass

    inertia = {
        "ixx": params["Ixx"] - mass * (y * y + z * z),
        "ixy": params["Ixy"] + mass * x * y,
        "ixz": params["Ixz"] + mass * x * z,
        "iyy": params["Iyy"] - mass * (x * x + z * z),
        "iyz": params["Iyz"] + mass * y * z,
        "izz": params["Izz"] - mass * (x * x + y * y),
    }

    # 很小的乘积惯量视为数值噪声。对角主惯量不做回退，正常写入换算结果。
    for key in ("ixy", "ixz", "iyz"):
        inertia[key] = zero_small_off_diagonal_inertia(inertia[key])

    return inertia


def update_body_inertial(body: ET.Element, params: dict[str, float]) -> None:
    """更新单个 MuJoCo body 的 inertial 参数。"""
    mass = params["m"]
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")

    # 一次矩除以质量，得到质心相对当前 body 坐标系的位置。
    com = (params["mx"] / mass, params["my"] / mass, params["mz"] / mass)

    inertial = ensure_child(body, "inertial")
    inertia = central_inertia(params)

    inertial.set("mass", fmt(mass))
    inertial.set("pos", " ".join(fmt(value) for value in com))

    # 使用 fullinertia 写入完整惯量张量。MuJoCo 的顺序为：
    # ixx iyy izz ixy ixz iyz
    inertial.set(
        "fullinertia",
        " ".join(
            fmt(inertia[key])
            for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )

    # diaginertia 和 fullinertia 不能同时使用。模板中原来是 diaginertia，
    # 改成完整惯量后需要删掉该属性。
    inertial.attrib.pop("diaginertia", None)

    # MuJoCo 不允许 fullinertia 与 inertial 的 quat 同时出现。
    # fullinertia 已经在当前 body 坐标系中给出完整惯量矩阵，因此删除 inertial
    # 自身的姿态属性。注意这不是 body 的 quat，不会改变连杆或关节几何位置。
    inertial.attrib.pop("quat", None)


def update_joint_dynamics(joint: ET.Element, params: dict[str, float]) -> None:
    """更新单个 MuJoCo joint 的摩擦和阻尼参数。

    MuJoCo 中：
    - frictionloss 表示库伦/静摩擦损失，对应 CSV 的 Fric1。
    - damping 表示粘性阻尼系数，对应 CSV 的 Fric2。
    """
    joint.set("frictionloss", fmt(abs(params["Fric1"])))
    joint.set("damping", fmt(abs(params["Fric2"])))


def rebuild_xml() -> None:
    """执行完整的 MuJoCo XML 重建流程。"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not SOURCE_XML.exists():
        raise FileNotFoundError(f"Source XML not found: {SOURCE_XML}")

    OUTPUT_XML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_XML, OUTPUT_XML)

    params_by_joint = load_identified_params(CSV_PATH)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(OUTPUT_XML, parser=parser)
    root = tree.getroot()

    for index, joint_column in enumerate(JOINT_COLUMNS):
        params = params_by_joint[joint_column]
        update_body_inertial(find_named(root, "body", BODY_NAMES[index]), params)
        update_joint_dynamics(find_named(root, "joint", MJCF_JOINT_NAMES[index]), params)

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {OUTPUT_XML.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    rebuild_xml()
