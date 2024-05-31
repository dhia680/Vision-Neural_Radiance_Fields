
import numpy as np
import torch
from config import *


def extract_origin_and_direction(pose_matrix, u, v, camera_intrinsics):
    """
    @brief This function extracts the camera pose, origin and direction of the pixel rays in world coordinates from the camera pose matrix and pixel coordinates in the image plane.

    The camera pose is represented by a 4x4 matrix, pose_matrix, which transforms points from world coordinates to camera coordinates.
    The origin $o$ of the camera in world coordinates is the translation component of the pose_matrix.
    The direction $d$ of the pixel rays in camera coordinates is calculated from the pixel coordinates in the image plane and the camera intrinsics. The direction of the pixel rays in world coordinates is calculated by transforming the direction of the pixel rays in camera coordinates to world coordinates using the rotation component of the pose_matrix.

    The image plane is assumed to be at z = f in camera coordinates, where f is the focal length of the camera. The pixel coordinates (u, v) are assumed to be in the image plane.
    

    @param pose_matrix: (B, 4, 4); camera pose matrix
    @param u: (im_height, im_width, B) or (B) or (im_height, im_wdith); pixel coordinates in the image plane
    @param v: (im_height, im_width, B) or (B) or (im_height, im_wdith); pixel coordinates in the image plane
    @return (im_height, im_width, B, 6) [o_world, d_world]
    
    """

    if u.ndim == 1 and v.ndim == 1: # a single pixel for all cameras
        u = u[None, None, :]
        v = v[None, None, :]
    elif u.ndim == 2 and v.ndim == 2: # all pixels for a single camera
        u = u[:, :, None].expand(-1, -1, pose_matrix.shape[0])
        v = v[:, :, None].expand(-1, -1, pose_matrix.shape[0])

    image_height, image_width, B = u.shape
    u = torch.tensor(u, device = device, dtype = dtype)
    v = torch.tensor(v, device = device, dtype = dtype)

    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    f = camera_intrinsics['f']

    # calculate phi, the angle of rotation about the x-axis of the pixel ray in camera coordinates
    phi = torch.atan2(v - cy, f * torch.ones_like(v))
    # calculate theta, the angle of rotation about the y-axis of the pixel ray in camera coordinates
    theta = torch.atan2(u - cx, f * torch.ones_like(v))

    o_world = pose_matrix[:, :3, 3] # B x 3; origin of the camera in world coordinates
    R = pose_matrix[:, :3, :3] # B x 3 x 3;  rotation matrix of the camera in world coordinates

    d_camera = torch.stack([torch.sin(phi), -torch.cos(phi) * torch.sin(theta), torch.cos(phi) * torch.cos(theta)], dim = 0) # 3 x im_height x im_width x B

    res = []
    for b in range(R.shape[0]):
        d_w = -torch.tensordot(R[b], d_camera[:, :, :, b], dims = 1) # 3 x im_height x im_width
        res.append(d_w)
    d_world = torch.stack(res, dim = 0) # B x 3 x im_height x im_width

    o_world = o_world[:, :, None, None].expand(-1, -1, image_height, image_width)
    concat = torch.cat([o_world, d_world], dim = 1) # B x 6 x im_height x im_width
    # => im_height x im_width x B x 6
    return concat.permute(2, 3, 0, 1) # im_height x im_width x B x 6

