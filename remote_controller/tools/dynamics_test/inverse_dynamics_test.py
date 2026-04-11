#!/usr/bin/env python3
"""
Mockway Robot Inverse Dynamics Test using Pinocchio

This script demonstrates inverse dynamics computation for the Mockway robot:
- Loads the URDF model
- Computes required joint torques given positions, velocities, and accelerations
- Tests with various trajectories
"""

import numpy as np
import pinocchio as pin
import os
import matplotlib.pyplot as plt
from pathlib import Path


class MockwayDynamics:
    """Wrapper class for Mockway robot dynamics using Pinocchio"""

    def __init__(self, urdf_path):
        """
        Initialize the dynamics model

        Args:
            urdf_path: Path to the URDF file
        """
        # Load the model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Get model information
        self.nq = self.model.nq  # Number of position variables
        self.nv = self.model.nv  # Number of velocity variables

        print(f"Model loaded: {self.model.name}")
        print(f"Number of joints: {self.nq}")
        print(f"Joint names: {[self.model.names[i] for i in range(1, self.model.njoints)]}")

    def compute_inverse_dynamics(self, q, v, a):
        """
        Compute inverse dynamics: tau = M(q)*a + C(q,v) + g(q)

        Args:
            q: Joint positions (nq,)
            v: Joint velocities (nv,)
            a: Joint accelerations (nv,)

        Returns:
            tau: Required joint torques (nv,)
        """
        tau = pin.rnea(self.model, self.data, q, v, a)
        return tau

    def compute_mass_matrix(self, q):
        """
        Compute the joint space inertia matrix M(q)

        Args:
            q: Joint positions (nq,)

        Returns:
            M: Mass matrix (nv, nv)
        """
        M = pin.crba(self.model, self.data, q)
        return M

    def compute_coriolis(self, q, v):
        """
        Compute Coriolis and centrifugal forces C(q,v)

        Args:
            q: Joint positions (nq,)
            v: Joint velocities (nv,)

        Returns:
            c: Coriolis forces (nv,)
        """
        c = pin.computeCoriolisMatrix(self.model, self.data, q, v) @ v
        return c

    def compute_gravity(self, q):
        """
        Compute gravity torques g(q)

        Args:
            q: Joint positions (nq,)

        Returns:
            g: Gravity torques (nv,)
        """
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return g

    def get_end_effector_pose(self, q):
        """
        Get end effector pose for a given joint configuration

        Args:
            q: Joint positions (nq,)

        Returns:
            position: 3D position of end effector
            orientation: Rotation matrix of end effector
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the last link's transformation
        ee_frame_id = self.model.nframes - 1
        ee_transform = self.data.oMf[ee_frame_id]

        return ee_transform.translation, ee_transform.rotation


def test_static_configurations():
    """Test inverse dynamics at static configurations"""
    print("\n" + "="*60)
    print("TEST 1: Static Configurations (Zero Velocity & Acceleration)")
    print("="*60)

    # Find URDF file
    workspace_dir = Path(__file__).parent.parent.parent
    urdf_path = workspace_dir / "mockway_description/urdf/mockway_description.urdf"

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Initialize dynamics
    dynamics = MockwayDynamics(str(urdf_path))

    # Test configurations (6-DOF robot)
    test_configs = {
        "Zero position": np.zeros(dynamics.nq),
        "Joint1 = 45°": np.array([np.pi/4, 0, 0, 0, 0, 0]),
        "Joint2 = 45°": np.array([0, np.pi/4, 0, 0, 0, 0]),
        "Joint3 = 45°": np.array([0, 0, np.pi/4, 0, 0, 0]),
        "All joints = 30°": np.array([np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6]),
        "Wrist config": np.array([0, 0, 0, np.pi/4, np.pi/4, np.pi/4]),
    }

    for name, q in test_configs.items():
        v = np.zeros(dynamics.nv)
        a = np.zeros(dynamics.nv)

        # Compute inverse dynamics
        tau = dynamics.compute_inverse_dynamics(q, v, a)

        # At static configuration, tau should equal gravity torques
        g = dynamics.compute_gravity(q)

        print(f"\n{name}:")
        print(f"  Joint positions: {np.rad2deg(q)} deg")
        print(f"  Required torques: {tau} Nm")
        print(f"  Gravity torques:  {g} Nm")
        print(f"  Difference: {np.linalg.norm(tau - g):.6e} Nm")


def test_trajectory_tracking():
    """Test inverse dynamics along a sinusoidal trajectory"""
    print("\n" + "="*60)
    print("TEST 2: Trajectory Tracking (Sinusoidal Motion)")
    print("="*60)

    # Find URDF file
    workspace_dir = Path(__file__).parent.parent.parent
    urdf_path = workspace_dir / "mockway_description/urdf/mockway_description.urdf"

    # Initialize dynamics
    dynamics = MockwayDynamics(str(urdf_path))

    # Generate sinusoidal trajectory
    duration = 5.0  # seconds
    dt = 0.01  # time step
    t = np.arange(0, duration, dt)

    # Trajectory parameters (6-DOF robot)
    amplitude = np.array([np.pi/3, np.pi/4, np.pi/6, np.pi/8, np.pi/8, np.pi/8])  # amplitude for each joint
    frequency = np.array([0.5, 0.7, 0.6, 0.8, 0.9, 1.0])  # Hz

    # Preallocate arrays
    n_steps = len(t)
    q_traj = np.zeros((n_steps, dynamics.nq))
    v_traj = np.zeros((n_steps, dynamics.nv))
    a_traj = np.zeros((n_steps, dynamics.nv))
    tau_traj = np.zeros((n_steps, dynamics.nv))

    # Generate trajectory
    for i, ti in enumerate(t):
        # Position: q = A * sin(2*pi*f*t)
        q_traj[i] = amplitude * np.sin(2 * np.pi * frequency * ti)

        # Velocity: v = A * 2*pi*f * cos(2*pi*f*t)
        v_traj[i] = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * ti)

        # Acceleration: a = -A * (2*pi*f)^2 * sin(2*pi*f*t)
        a_traj[i] = -amplitude * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * ti)

        # Compute inverse dynamics
        tau_traj[i] = dynamics.compute_inverse_dynamics(q_traj[i], v_traj[i], a_traj[i])

    # Print statistics
    print(f"\nTrajectory duration: {duration} s")
    print(f"Number of samples: {n_steps}")
    print(f"\nJoint torque statistics:")
    for j in range(dynamics.nv):
        print(f"  Joint {j+1}:")
        print(f"    Mean:   {np.mean(tau_traj[:, j]):.4f} Nm")
        print(f"    Max:    {np.max(tau_traj[:, j]):.4f} Nm")
        print(f"    Min:    {np.min(tau_traj[:, j]):.4f} Nm")
        print(f"    Std:    {np.std(tau_traj[:, j]):.4f} Nm")

    # Plot results
    plot_trajectory_results(t, q_traj, v_traj, a_traj, tau_traj)

    return t, q_traj, v_traj, a_traj, tau_traj


def test_dynamics_decomposition():
    """Test decomposition of dynamics into M, C, and g components"""
    print("\n" + "="*60)
    print("TEST 3: Dynamics Decomposition (M*a + C*v + g)")
    print("="*60)

    # Find URDF file
    workspace_dir = Path(__file__).parent.parent.parent
    urdf_path = workspace_dir / "mockway_description/urdf/mockway_description.urdf"

    # Initialize dynamics
    dynamics = MockwayDynamics(str(urdf_path))

    # Test configuration (6-DOF robot)
    q = np.array([np.pi/4, np.pi/6, -np.pi/8, np.pi/12, -np.pi/12, np.pi/6])
    v = np.array([0.5, -0.3, 0.2, -0.1, 0.15, -0.25])
    a = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.4])

    print(f"\nConfiguration:")
    print(f"  q = {np.rad2deg(q)} deg")
    print(f"  v = {v} rad/s")
    print(f"  a = {a} rad/s²")

    # Compute using RNEA (Recursive Newton-Euler Algorithm)
    tau_rnea = dynamics.compute_inverse_dynamics(q, v, a)

    # Compute components separately
    M = dynamics.compute_mass_matrix(q)
    c = dynamics.compute_coriolis(q, v)
    g = dynamics.compute_gravity(q)

    # Reconstruct tau = M*a + c + g
    tau_reconstructed = M @ a + c + g

    print(f"\nMass Matrix M(q):")
    print(M)

    print(f"\nCoriolis forces c(q,v):")
    print(c)

    print(f"\nGravity torques g(q):")
    print(g)

    print(f"\nInverse Dynamics Results:")
    print(f"  tau (RNEA):          {tau_rnea} Nm")
    print(f"  tau (M*a + c + g):   {tau_reconstructed} Nm")
    print(f"  Difference:          {np.linalg.norm(tau_rnea - tau_reconstructed):.6e} Nm")


def plot_trajectory_results(t, q, v, a, tau):
    """Plot trajectory tracking results"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Inverse Dynamics - Trajectory Tracking Results (6-DOF Robot)', fontsize=14)

    n_joints = q.shape[1]
    joint_names = [f'Joint {i+1}' for i in range(n_joints)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 6 distinct colors

    # Plot positions
    axes[0].set_title('Joint Positions')
    for i in range(n_joints):
        axes[0].plot(t, np.rad2deg(q[:, i]), color=colors[i], label=joint_names[i], linewidth=1.5)
    axes[0].set_ylabel('Position [deg]')
    axes[0].legend(ncol=3, loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot velocities
    axes[1].set_title('Joint Velocities')
    for i in range(n_joints):
        axes[1].plot(t, v[:, i], color=colors[i], label=joint_names[i], linewidth=1.5)
    axes[1].set_ylabel('Velocity [rad/s]')
    axes[1].legend(ncol=3, loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Plot accelerations
    axes[2].set_title('Joint Accelerations')
    for i in range(n_joints):
        axes[2].plot(t, a[:, i], color=colors[i], label=joint_names[i], linewidth=1.5)
    axes[2].set_ylabel('Acceleration [rad/s²]')
    axes[2].legend(ncol=3, loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Plot torques
    axes[3].set_title('Required Joint Torques (Inverse Dynamics)')
    for i in range(n_joints):
        axes[3].plot(t, tau[:, i], color=colors[i], label=joint_names[i], linewidth=1.5)
    axes[3].set_ylabel('Torque [Nm]')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(ncol=3, loc='upper right', fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent
    output_path = output_dir / "trajectory_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("Mockway Robot - Inverse Dynamics Testing with Pinocchio")
    print("="*60)

    try:
        # Run tests
        test_static_configurations()
        test_dynamics_decomposition()
        test_trajectory_tracking()

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
