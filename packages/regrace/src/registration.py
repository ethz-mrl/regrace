import numpy as np
import open3d as o3d
import torch

from .data import Data
from .dataset import Dataset


class RANSAC():

    @staticmethod
    def make_open3d_feature(data: torch.Tensor, dim: int, npts: int):
        # convert to open3d format
        feature = o3d.pipelines.registration.Feature()
        feature.resize(dim, npts)
        feature.data = data.cpu().numpy().astype('d').transpose()
        return feature

    @staticmethod
    def make_open3d_point_cloud(xyz: torch.Tensor,
                                color=None,
                                voxel_size=None):
        # convert to open3d format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            pcd.paint_uniform_color(color)
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size)
        return pcd

    @staticmethod
    def get_ransac_result(
            feat1: torch.Tensor,
            feat2: torch.Tensor,
            pos1: torch.Tensor,
            pos2: torch.Tensor,
            ransac_dist_th=1,
            ransac_max_it=10000,
            ransac_n_inliers=3
    ) -> o3d.pipelines.registration.RegistrationResult:
        # code by Joshua Knights (CSIRO)
        # convert to open3d format
        feature_dim = feat1.shape[1]
        pcd_feat1 = RANSAC.make_open3d_feature(feat1, feature_dim,
                                               feat1.shape[0])
        pcd_feat2 = RANSAC.make_open3d_feature(feat2, feature_dim,
                                               feat2.shape[0])
        pcd_coord1 = RANSAC.make_open3d_point_cloud(pos1.numpy())
        pcd_coord2 = RANSAC.make_open3d_point_cloud(pos2.numpy())

        # ransac based eval
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_coord1,
            pcd_coord2,
            pcd_feat1,
            pcd_feat2,
            mutual_filter=True,
            max_correspondence_distance=ransac_dist_th,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(False),
            ransac_n=ransac_n_inliers,
            checkers=[
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnDistance(ransac_dist_th)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                ransac_max_it, 0.999))


class ICP():

    @staticmethod
    def get_icp_result(
            pos1: torch.Tensor,
            pos2: torch.Tensor,
            ransac_result: o3d.pipelines.registration.RegistrationResult,
            icp_dist_th=0.5,
            icp_max_it=10000) -> o3d.pipelines.registration.RegistrationResult:
        pcd_coord1 = RANSAC.make_open3d_point_cloud(pos1.numpy())
        pcd_coord2 = RANSAC.make_open3d_point_cloud(pos2.numpy())

        # icp based eval
        return o3d.pipelines.registration.registration_icp(
            pcd_coord1, pcd_coord2, icp_dist_th, ransac_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=icp_max_it))


def register(data_1: Data,
             data_2: Data,
             node_features_1: torch.Tensor,
             node_positions_1: torch.Tensor,
             node_features_2: torch.Tensor,
             node_positions_2: torch.Tensor,
             normalization_constant: float,
             eval_idx: int,
             compare_idx: int,
             verbose: bool = False,
             return_errors: bool = False) -> bool | tuple[bool, float, float]:
    # coarse registration
    ransac_result = RANSAC().get_ransac_result(
        feat1=node_features_1,
        feat2=node_features_2,
        pos1=node_positions_1 * normalization_constant,
        pos2=node_positions_2 * normalization_constant)
    if verbose:
        print(
            f"RANSAC inliers: {len(ransac_result.correspondence_set)} >> {np.asarray(ransac_result.correspondence_set)}"
        )
        print(f"RANSAC fitness: {ransac_result.fitness:.2f}")
        print(f"RANSAC rsme: {ransac_result.inlier_rmse:.2f}")
    # get ransac inliers
    if len(ransac_result.correspondence_set) == 0:
        if return_errors:
            return False, 0.0, 0.0
        return False
    valid_idx1 = np.asarray(ransac_result.correspondence_set)[:, 0].astype(int)
    valid_idx2 = np.asarray(ransac_result.correspondence_set)[:, 1].astype(int)
    # get all points for ICP
    node_points1, node_center1, _ = data_1.to_collate_format(
        normalization_constant, False)
    node_points2, node_center2, _ = data_2.to_collate_format(
        normalization_constant, False)
    # get valid points for ICP
    inliner_node_points1 = [node_points1[i] for i in valid_idx1]
    inliner_node_points2 = [node_points2[i] for i in valid_idx2]
    inliner_node_center1 = torch.stack([node_center1[i] for i in valid_idx1])
    inliner_node_center2 = torch.stack([node_center2[i] for i in valid_idx2])
    # get center for ICP
    inliner_node_points1 = torch.cat([
        points[:, :3] + center
        for points, center in zip(inliner_node_points1, inliner_node_center1)
    ],
                                     dim=0) * normalization_constant
    inliner_node_points2 = torch.cat([
        points[:, :3] + center
        for points, center in zip(inliner_node_points2, inliner_node_center2)
    ],
                                     dim=0) * normalization_constant
    # fine registration
    icp_result = ICP().get_icp_result(pos1=inliner_node_points1,
                                      pos2=inliner_node_points2,
                                      ransac_result=ransac_result)
    # get full transformation
    inverse_pose_2 = np.zeros((4, 4))
    inverse_pose_2[:3, :3] = np.linalg.inv(data_2.pose[:3, :3])
    inverse_pose_2[:3, 3] = -np.linalg.inv(data_2.pose[:3, :3]).dot(
        data_2.pose[:3, 3])
    inverse_pose_2[3, 3] = 1
    full_transformation = np.dot(inverse_pose_2, data_1.pose)
    # get eugler angles from rotation matrix
    translation_error = np.linalg.norm(icp_result.transformation[:3, 3] -
                                       full_transformation[:3, 3])
    cosine_rotation_error = (
        np.trace(icp_result.transformation[:3, :3].copy().transpose(1, 0)
                 @ full_transformation[:3, :3]) - 1.) / 2.
    rotation_error = np.arccos(
        np.clip(cosine_rotation_error, a_min=-1 + 1e-6,
                a_max=1. - 1e-6)) * 180. / np.pi
    # check if the registration is successful
    if translation_error <= 2.0 and rotation_error <= 5.0:
        print(f"Registration successful: {eval_idx} -> {compare_idx}, "
              f"translation error: {translation_error:.2f}, "
              f"rotation error: {rotation_error:.2f}") if verbose else None
        success = True
    else:
        success = False
    # return errors if requested
    if return_errors:
        return success, float(translation_error), float(rotation_error)
    return success

def compute_consistency(
    node_features_1: torch.Tensor,
    node_positions_1: torch.Tensor,
    node_features_2: torch.Tensor,
    node_positions_2: torch.Tensor,
    normalization_constant: float,
    d_tresh: float = 5.0,
) -> float:
    # compute inliers
    ransac_result = RANSAC().get_ransac_result(
        feat1=node_features_1,
        feat2=node_features_2,
        pos1=node_positions_1 * normalization_constant,
        pos2=node_positions_2 * normalization_constant,
        ransac_n_inliers=4)

    # get the accumulated consistency
    inliers = np.asarray(ransac_result.correspondence_set)
    consistency = 0
    for i in range(len(inliers)):
        for j in range(i + 1, len(inliers)):
            dist_in_1 = np.linalg.norm(
                node_positions_1[inliers[i][0]].numpy() -
                node_positions_1[inliers[j][0]].numpy())
            dist_in_2 = np.linalg.norm(
                node_positions_2[inliers[i][1]].numpy() -
                node_positions_2[inliers[j][1]].numpy())
            consistency += max(
                1 - pow(np.abs(dist_in_1 - dist_in_2), 2) / pow(d_tresh, 2), 0)
    return consistency
