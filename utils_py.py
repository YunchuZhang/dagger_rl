import os
import numpy as np
import pickle
import sys
import os
import glob
import scipy.misc
import tensorflow as tf
import hyperparams_new as hyp

EPS = 1e-6

# Copied these parameters from MUJOCO_OFFLINE

XMIN = -0.5 # right (neg is left)
XMAX = 0.5 # right
YMIN = -0.5 # down (neg is up)
YMAX = 0.5 # down
ZMIN = 0.3 # forward
ZMAX = 1.3 # forward
FLOOR = 0.0 # ground (parallel with refcam)
CEIL = (FLOOR-0.5)

def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_rt(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4 
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def split_rt(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3])
    return r, t

def eul2rotm(rx, ry, rz):
    # copy of matlab, but order of inputs is different
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    sinz = np.sin(rz)
    siny = np.sin(ry)
    sinx = np.sin(rx)
    cosz = np.cos(rz)
    cosy = np.cos(ry)
    cosx = np.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = np.stack([r11,r12,r13],axis=-1)
    r2 = np.stack([r21,r22,r23],axis=-1)
    r3 = np.stack([r31,r32,r33],axis=-1)
    r = np.stack([r1,r2,r3],axis=0)
    return r

def apply_4x4(RT, XYZ):
    # RT is 4 x 4
    # XYZ is N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=1)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=1)
    # XYZ1 is N x 4

    XYZ1_t = np.transpose(XYZ1)
    # this is 4 x N

    XYZ2_t = np.dot(RT, XYZ1_t)
    # this is 4 x N
    
    XYZ2 = np.transpose(XYZ2_t)
    # this is N x 4
    
    XYZ2 = XYZ2[:,:3]
    # this is N x 3
    
    return XYZ2

def split_intrinsics(K):
    # K is 3 x 4 or 4 x 4
    fx = K[0,0]
    fy = K[1,1]
    x0 = K[0,2]
    y0 = K[1,2]
    return fx, fy, x0, y0
                    
def merge_intrinsics(fx, fy, x0, y0):
    # inputs are shaped []
    K = np.eye(4)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = x0
    K[1,2] = y0
    # K is shaped 4 x 4
    return K
                            
def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx *= sx
    fy *= sy
    x0 *= sx
    y0 *= sy
    return merge_intrinsics(fx, fy, x0, y0)

def Pixels2Camera(x,y,Z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, Z is a depth image in meters
    # their shapes are H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns XYZ, sized [B,H*W,3]
    
    H, W = Z.shape
    
    fx = np.reshape(fx, [1,1])
    fy = np.reshape(fy, [1,1])
    x0 = np.reshape(x0, [1,1])
    y0 = np.reshape(y0, [1,1])
    
    # unproject
    X = ((Z+EPS)/fx)*(x-x0)
    Y = ((Z+EPS)/fy)*(y-y0)
    
    X = np.reshape(X, [-1])
    Y = np.reshape(Y, [-1])
    Z = np.reshape(Z, [-1])
    XYZ = np.stack([X,Y,Z], axis=1)
    return XYZ

def depth2pointcloud(z, pix_T_cam):
    H = z.shape[0]
    W = z.shape[1]
    y, x = meshgrid2D(H, W)
    z = np.reshape(z, [H, W])
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def meshgrid2D(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    # outputs are Y x X
    return grid_y, grid_x

def gridcloud3D(Y, X, Z):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    z_ = np.linspace(0, Z-1, Z)
    y, x, z = np.meshgrid(y_, x_, z_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    z = np.reshape(z, [-1])
    xyz = np.stack([x,y,z], axis=1).astype(np.float32)
    return xyz

def gridcloud2D(Y, X):
    x_ = np.linspace(0, X-1, X)
    y_ = np.linspace(0, Y-1, Y)
    y, x = np.meshgrid(y_, x_, indexing='ij')
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    xy = np.stack([x,y], axis=1).astype(np.float32)
    return xyz

def normalize(im):
    im = im - np.min(im)
    im = im / np.max(im)
    return im

# def normalize_each(ims):
#     outs = []
#     for im in ims:
#         outs.append(normalize(im))
#     outs = np.stack(outs, axis=0)
#     return outs

def assert_exists(path):
    if not os.path.exists(path):
        print('%s does not exist' % path)
        assert(False)

def load_from_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data

def vis_depth(depth, maxdepth=80.0, log_vis=True):
    depth[depth<=0.0] = maxdepth
    if log_vis:
        depth = np.log(depth)
        depth = np.clip(depth, 0, np.log(maxdepth))
    else:
        depth = np.clip(depth, 0, maxdepth)
    depth = (depth*255.0).astype(np.uint8)
    return depth
    # # print '%.2f, %.2f, %.2f' % (np.min(depth_im), np.mean(depth_im), np.max(depth_im))
    # depth_im = np.clip(depth_im, 0, 0.5)
    # # print '%.2f, %.2f, %.2f' % (np.min(depth_im), np.mean(depth_im), np.max(depth_im))
    # depth_im[depth_im==0] = np.max(depth_im)
    # depth_im = utils_py.normalize(depth_im)
    # # print '%.2f, %.2f, %.2f' % (np.min(depth_im), np.mean(depth_im), np.max(depth_im))
    # scipy.misc.imsave('%05d_rgb_cam%d.png' % (frame_ind, cam_ind), rgb)
    # scipy.misc.imsave('%05d_depth_cam%d.png' % (frame_ind, cam_ind), depth_im)
    # print 'saved rgb and depth'

def print_stats(tensor, name):
    print('%s min = %.2f, mean = %.2f, max = %.2f' % (name, np.min(tensor), np.mean(tensor), np.max(tensor)))

def deg2rad(deg):
    return np.deg2rad(deg)
def rad2deg(rad):
    return np.rad2deg(rad)
def Ref2Mem(xyz, MH, MW, MD):
    # xyz is N x 3, in ref coordinates
    # transforms ref coordinates into mem coordinates
    N, C = xyz.shape
    assert(C==3)
    mem_T_ref = get_mem_T_ref(MH, MW, MD)
    xyz = apply_4x4(mem_T_ref, xyz)
    return xyz

# def Mem2Ref(xyz_mem, MH, MW, MD):
#     # xyz is B x N x 3, in mem coordinates
#     # transforms mem coordinates into ref coordinates
#     B, N, C = xyz_mem.get_shape().as_list()
#     ref_T_mem = get_ref_T_mem(B, MH, MW, MD)
#     xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
#     return xyz_ref

def get_mem_T_ref(MH, MW, MD):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[0,3] = -XMIN
    center_T_ref[1,3] = -YMIN
    center_T_ref[2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(MW)
    VOX_SIZE_Y = (YMAX-YMIN)/float(MH)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(MD)
    
    # scaling
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0,0] = 1./VOX_SIZE_X
    mem_T_center[1,1] = 1./VOX_SIZE_Y
    mem_T_center[2,2] = 1./VOX_SIZE_Z
    
    mem_T_ref = np.dot(mem_T_center, center_T_ref)
    return mem_T_ref

def get_ref_T_mem(MH, MW, MD):
    mem_T_ref = get_mem_T_ref(MH, MW, MD)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = np.linalg.inv(mem_T_ref)
    return ref_T_mem

def voxelize_xyz(xyz_ref, proto):
    # proto shows the size of the voxelgrid
    MH, MW, MD = proto.shape
    xyz_mem = Ref2Mem(xyz_ref, MH, MW, MD)
    # this is B x N x 3
    # x, y, z = tf.split(xyz_mem, 3, axis=2)
    voxels = get_occupancy(xyz_mem, proto)
    voxels = np.reshape(voxels, [MH, MW, MD, 1])
    return voxels

def get_inbounds(XYZ, proto, already_mem=False):
    # XYZ is H*W x 3
    # proto is MH x MW x MD
    MH, MW, MD = proto.shape
    
    if not already_mem:
        XYZ = Ref2Mem(XYZ, MH, MW, MD)
    
    x_valid = np.logical_and(
        np.greater_equal(XYZ[:,0], 0.0), 
        np.less(XYZ[:,0], float(MW)-1.0))
    y_valid = np.logical_and(
        np.greater_equal(XYZ[:,1], 0.0), 
        np.less(XYZ[:,1], float(MH)-1.0))
    z_valid = np.logical_and(
        np.greater_equal(XYZ[:,2], 0.0), 
        np.less(XYZ[:,2], float(MD)-1.0))
    inbounds = np.logical_and(np.logical_and(x_valid, y_valid), z_valid)
    return inbounds

def sub2ind3D_zyx(depth, height, width, d, h, w):
    # same as sub2ind3D, but inputs in zyx order
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def sub2ind3D_yxz(height, width, depth, h, w, d):
    return h*width*depth + w*depth + d

def get_occupancy(XYZ, proto):
    # XYZ is N x 3
    # we want to fill a voxel tensor with 1's at these inds

    MH, MW, MD = proto.shape

    inbounds = get_inbounds(XYZ, proto, already_mem=True)
    inds = np.where(inbounds)

    XYZ = np.reshape(XYZ[inds], [-1, 3])
    # XYZ is N x 3

    # this is more accurate than a cast/floor, but runs into issues when MH==0
    XYZ = np.round(XYZ).astype(np.int32)
    x = XYZ[:,0]
    y = XYZ[:,1]
    z = XYZ[:,2]

    voxels = np.zeros([MH, MW, MD], np.float32)
    voxels[y, x, z] = 1.0

    return voxels

def wrap2pi(rad_angle):
    # rad_angle can be any shape
    # puts the angle into the range [-pi, pi]
    return np.arctan2(np.sin(rad_angle), np.cos(rad_angle))

def convert_occ_to_height(occ):
    H, W, D, C = occ.shape
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    height = np.linspace(float(H), 1.0, H)
    height = np.reshape(height, [1, H, 1, 1, 1])
    height = np.max(occ*height, axis=1)/float(D)
    height = np.reshape(height, [W, D, C])
    return height


def create_depth_image(xy, Z, H, W):

    # turn the xy coordinates into image inds
    xy = np.round(xy)

    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (Z > 0)
    valid = (xy[:,0] < W-1) & (xy[:,1] < H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (Z[:] > 0)

    # gather these up
    xy = xy[valid]
    Z = Z[valid]
    
    inds = sub2ind(H,W,xy[:,1],xy[:,0])
    depth = np.zeros((H*W), np.float32)

    for (index, replacement) in zip(inds, Z):
        depth[index] = replacement
    depth[np.where(depth == 0.0)] = 70.0
    depth = np.reshape(depth, [H, W])

    return depth
