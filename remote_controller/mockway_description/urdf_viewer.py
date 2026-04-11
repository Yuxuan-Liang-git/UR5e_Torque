#!/usr/bin/env python3
import sys
import numpy as np
import pinocchio as pin
from pathlib import Path

def view_urdf():
    # 获取 URDF 路径
    urdf_path = Path(__file__).parent / "urdf" / "mockway_description.urdf"
    mesh_dir = Path(__file__).parent / "meshes"
    
    if not urdf_path.exists():
        print(f"错误: 找不到 URDF 文件 {urdf_path}")
        return

    try:
        import meshcat
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError:
        print("错误: 请先安装 meshcat。使用命令: pip install meshcat")
        return

    # 加载模型
    # 注意：如果环境中没有设置 ROS_PACKAGE_PATH，Pinocchio 可能找不到 package:// 路径
    # 我们在这里手动处理一下 mesh 的加载路径，或者依赖环境中已有的 ROS 环境
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        str(urdf_path), 
        package_dirs=[str(Path(__file__).parent.parent)] # 搜索上一级目录以匹配 package://mockway_description
    )

    # 打印关节信息，验证 0 位姿
    print("\n" + "="*40)
    print(f"模型名称: {model.name}")
    print(f"自由度 (nq): {model.nq}")
    print("关节列表及 0 位状态:")
    for i in range(1, model.njoints):
        print(f"  关节 {i}: {model.names[i]}")
    print("="*40 + "\n")

    # 启动 Meshcat 可视化
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    
    try:
        viz.initViewer(open=True)
    except Exception as e:
        print(f"启动 Viewer 失败: {e}")
        return

    viz.loadViewerModel()

    # 初始化位姿
    q = pin.neutral(model)
    viz.display(q)

    # 简易的手动控制循环（在控制台输入角度）
    print("\n" + "="*40)
    print("手动控制模式:")
    print("输入格式: 'joint_index angle_in_degrees' (例如 '2 45')")
    print("输入 'reset' 重置到 0 位，输入 'q' 退出。")
    print("="*40)

    try:
        import time
        while True:
            user_input = input("\n请输入指令: ").strip().lower()
            if user_input == 'q':
                break
            elif user_input == 'reset':
                q = pin.neutral(model)
                viz.display(q)
                print("已重置到 0 位姿")
                continue
            
            try:
                parts = user_input.split()
                if len(parts) == 2:
                    idx = int(parts[0]) - 1 # 转换成从 0 开始的索引
                    angle = np.deg2rad(float(parts[1]))
                    
                    if 0 <= idx < model.nq:
                        q[idx] = angle
                        viz.display(q)
                        print(f"设置关节 {idx+1} 为 {parts[1]}°")
                    else:
                        print(f"错误: 关节索引超出范围 (1-{model.nq})")
                else:
                    print("格式错误。例如: '2 45'")
            except ValueError:
                print("无效的输入")
                
    except KeyboardInterrupt:
        print("\n正在关闭...")

if __name__ == "__main__":
    view_urdf()
