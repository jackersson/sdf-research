import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree


def compute_trimesh_chamfer(gt_points: trimesh.points.PointCloud,
                            gen_mesh: trimesh.base.Trimesh,
                            offset,
                            scale,
                            num_mesh_samples=30000):
    """
    Initial implementation:
    https://github.com/facebookresearch/DeepSDF/blob/master/deep_sdf/metrics/chamfer.py

    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    :param gt_points: of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    :param gen_mesh: output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(
        gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer
