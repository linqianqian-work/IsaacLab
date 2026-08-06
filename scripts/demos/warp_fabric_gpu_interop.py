# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduce pre-Fabric CUDA LBVH corruption with a large Warp mesh.

The default LBVH run can make parts of the Replicator kitchen disappear or render at
stale poses:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/warp_fabric_gpu_interop.py --viz kit

Using a CUDA SAH BVH for the same mesh avoids the issue:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/warp_fabric_gpu_interop.py --viz kit --bvh_constructor sah
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--bvh_constructor",
    choices=("lbvh", "sah"),
    default="lbvh",
    help="Warp BVH constructor. LBVH reproduces the issue; SAH avoids it.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of simulation steps to run; zero runs until the app is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path


def kitchen_usd_path() -> str:
    """Return the Replicator kitchen asset used by the Arena reproduction."""
    staging_root = ISAACLAB_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")
    return f"{staging_root}/Arena/assets/background_library/replicator_kitchen/kitchen_peninsula.usda"


def extract_kitchen_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Extract the complete kitchen mesh as Arena did for its background collision proxy."""
    stage = Usd.Stage.Open(retrieve_file_path(kitchen_usd_path()))
    assert stage is not None, "Failed to open the Replicator kitchen USD."
    vertices_by_prim: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        prim_path = str(prim.GetPath())
        if prim_path.endswith("/KitchenRoom/Floor") or prim_path.endswith("/CapSurface_02"):
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get()
        face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
        face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            continue

        world_transform = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        )
        local_vertices = np.asarray(points, dtype=np.float64)
        homogeneous_vertices = np.column_stack([local_vertices, np.ones(len(local_vertices))])
        vertices_by_prim.append((homogeneous_vertices @ world_transform)[:, :3])

        index_offset = 0
        for face_count in face_counts:
            for triangle_index in range(1, face_count - 1):
                triangles.append(
                    (
                        face_indices[index_offset] + vertex_offset,
                        face_indices[index_offset + triangle_index] + vertex_offset,
                        face_indices[index_offset + triangle_index + 1] + vertex_offset,
                    )
                )
            index_offset += face_count
        vertex_offset += len(local_vertices)

    assert vertices_by_prim and triangles, "The Replicator kitchen contained no extractable mesh geometry."
    return np.vstack(vertices_by_prim).astype(np.float32), np.asarray(triangles, dtype=np.int32)


def create_pre_fabric_warp_mesh() -> tuple[wp.Mesh, wp.array, wp.array]:
    """Construct and retain the large CUDA BVH before PhysX/Fabric initialization."""
    assert str(args_cli.device).startswith("cuda"), "This reproducer requires a CUDA simulation device."
    vertices, triangles = extract_kitchen_mesh()
    points = wp.array(vertices, dtype=wp.vec3, device=args_cli.device)
    indices = wp.array(triangles.reshape(-1), dtype=wp.int32, device=args_cli.device)
    mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=args_cli.bvh_constructor)
    wp.synchronize_device(args_cli.device)
    print(
        f"[repro] Retaining pre-Fabric CUDA {args_cli.bvh_constructor.upper()} BVH: "
        f"{len(vertices)} vertices, {len(triangles)} triangles."
    )
    return mesh, points, indices


def spawn_scene() -> None:
    """Spawn the affected kitchen."""
    kitchen_cfg = sim_utils.UsdFileCfg(usd_path=kitchen_usd_path())
    kitchen_cfg.func("/World/Kitchen", kitchen_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)


def run_simulator(sim: SimulationContext) -> None:
    """Step and render the scene."""
    step = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        sim.step(render=True)
        step += 1


def main() -> None:
    """Build the Warp BVH before creating the Fabric simulation."""
    # Keep the cache alive for the lifetime of the simulation, matching Arena's original placement pool.
    warp_cache = create_pre_fabric_warp_mesh()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, use_fabric=True)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((1.8, 0.8, 1.6), (0.0, 0.4, 0.9))
    spawn_scene()
    sim.reset()
    print(f"[repro] Scene ready with {args_cli.bvh_constructor.upper()}. Watch for missing or displaced kitchen parts.")
    run_simulator(sim)
    del warp_cache


if __name__ == "__main__":
    main()
    simulation_app.close()
