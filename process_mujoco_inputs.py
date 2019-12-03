import imageio
import numpy as np
import hyperparams as hyp
import utils_py
import tensorflow as tf
import os

def get_origin_T_cam_from_angles(distance,angles, do_print=False):
    # the camera is somewhere along a sphere with radius 0.7
    # the position is specified by two angles:
    # elevation and azimuth, both in degrees

    el, az = angles

    az = np.deg2rad(az)
    el = np.deg2rad(el)

    # i can figure out the cam position using trigonometry
    dist = distance
    z = dist*np.cos(el)*np.cos(az)
    x = dist*np.cos(el)*np.sin(az)
    y = -dist*np.sin(el)

    if do_print:
        print('az, el = %.2f, %.2f; \tx, y, z = %.2f, %.2f, %.2f' % (
            np.rad2deg(az), np.rad2deg(el), x, y, z))

    # the cam angles are more-or-less done already
    rx = -(np.pi-el)
    ry = az
    rz = 0.0

    r = utils_py.eul2rotm(rx, ry, rz)
    t = np.stack([x,y,z])

    rt = utils_py.merge_rt(r, t)
    # almost done, except it seems mujo's y axis points up
    # so we are not quite in cam coords

    mujo_T_cam = np.eye(4, dtype=np.float32)
    mujo_T_cam[1,1] = -1.0
    cam_T_mujo = np.linalg.inv(mujo_T_cam)

    # we want both sides of the rt to be in cam coords
    origin_T_cam = np.matmul(np.matmul(cam_T_mujo, rt), mujo_T_cam)
    return origin_T_cam

def get_inputs(pkl, puck_z):
    all_rgbs = pkl['image_observation']
    all_depths = pkl['depth_observation']
    all_cam_info = pkl['cam_info_observation']
    state_position = pkl['observation_with_orientation']

    # all_rgbs is B' x S' x H x W x 3
    # all_depths is B' x S' x H x W x 3
    S, H, W = hyp.S, hyp.H, hyp.W


    all_rgbs = np.reshape(all_rgbs,[all_rgbs.shape[0], S, H, W, 3])
    all_depths = np.reshape(all_depths, [all_depths.shape[0], S, H, W, 1])
    all_cam_info = np.reshape(all_cam_info, [all_cam_info.shape[0], S, -1])
    state_position = np.reshape(state_position, [state_position.shape[0], -1])


    datalen = all_rgbs.shape[0]
    seqlen = all_rgbs.shape[1]

    assert(H==all_rgbs.shape[2])
    assert(W==all_rgbs.shape[3])
    assert(H==all_depths.shape[2])
    assert(W==all_depths.shape[3])
    # or we can resize

    seq_inds = np.random.permutation(seqlen)[:hyp.S]
    B = datalen

    rgbs = all_rgbs[:, seq_inds]
    depths = all_depths[:, seq_inds]
    cam_angles = all_cam_info[:, seq_inds, :2]
    puck_position = state_position[:,-11:-9]
    puck_rotation = state_position[:,-9:]
    cam_dist = all_cam_info[:, seq_inds, 2]
    world_center = all_cam_info[:, seq_inds, 3:]
    world_center = world_center[0,0,:3].reshape([-1])

    rgbs = np.reshape(rgbs, [B, S, H, W, 3])
    depths = np.reshape(depths, [B, S, H, W, 1])
    cam_angles = np.reshape(cam_angles, [B, S, 2])
    puck_position = np.reshape(puck_position,[B, 2])
    puck_T_mujoco = np.reshape(puck_rotation,[B, 3, 3])

    cam_dist = np.reshape(cam_dist,[B, S, 1])

    # odd but ok
    depths = depths*2.0

    # judging from the pictures, the refcam i want has el=-180, az=90
    # used for fixed camera
    origin_T_camR = get_origin_T_cam_from_angles(1, (-180.0, 90.0), do_print=False)
    origin_T_camRs = np.reshape(origin_T_camR, [1, 1, 4, 4])
    origin_T_camRs = np.tile(origin_T_camRs, [B, S, 1, 1])

    # world origin to space origin transform (Rotation inferred from reference)
    mat = np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]])
    pos = np.array([world_center[1], world_center[2], world_center[0]])
    origin_T_mujoco = np.zeros((4, 4), dtype=float)
    origin_T_mujoco[:3, :3] = mat
    origin_T_mujoco[3][3] = 1
    origin_T_mujoco[:3, 3] = pos

    origin_T_camXs = np.zeros([B, S, 4, 4])
    camR_T_puck = np.zeros_like(puck_T_mujoco)


    for b in list(range(B)):
        for s in list(range(S)):
            origin_T_camXs[b,s] = get_origin_T_cam_from_angles(cam_dist[b,s],
                cam_angles[b,s], do_print=False)
        origin_T_puck = np.dot(origin_T_mujoco[:3,:3], np.linalg.inv(puck_T_mujoco[b]))
        camR_T_puck[b] = np.dot(np.linalg.inv(origin_T_camR)[:3,:3], origin_T_puck)

    FOV = 41.6
    focal_px = (W * 0.5) / np.tan(FOV * 0.5 * np.pi/180)
    fx = focal_px
    fy = focal_px
    x0 = W/2.0
    y0 = H/2.0
    pix_T_cam = utils_py.merge_intrinsics(fx, fy, x0, y0)
    pix_T_cams = np.reshape(pix_T_cam, [1, 1, 4, 4])
    pix_T_cams = np.tile(pix_T_cams, [B, S, 1, 1])

    puck_position = np.hstack((puck_position, np.reshape(puck_z,[B,1])))
    puck_xyz_origin = utils_py.apply_4x4(origin_T_mujoco, puck_position)
    puck_xyz_camR = utils_py.apply_4x4(np.linalg.inv(origin_T_camR), puck_xyz_origin)
    puck_xyz_camRs = np.expand_dims(puck_xyz_camR, axis=1)
    camRs_T_puck = np.expand_dims(camR_T_puck, axis=1)

    # we need pointclouds, not depthmaps
    V = hyp.V # max number of points in the pointcloud
    xyz_cams = np.zeros([B, S, V, 3])
    for b in list(range(B)):
        for s in list(range(S)):
            depth = depths[b,s]
            pix_T_cam = pix_T_cams[b,s]
            xyz_cam = utils_py.depth2pointcloud(depth, pix_T_cam)
            # throw away the junk
            xyz_cam = xyz_cam[xyz_cam[:,2] > 0.01]
            xyz_cam = xyz_cam[xyz_cam[:,2] < 5.0]

            np.random.shuffle(xyz_cam)
            if xyz_cam.shape[0] > V:
                xyz_cam = xyz_cam[0:V]
            elif xyz_cam.shape[0] < V:
                xyz_cam = np.pad(xyz_cam, [(0, V - xyz_cam.shape[0]), (0,0)],
                                 mode='constant', constant_values=0)
            xyz_cams[b, s] = xyz_cam

            camR_T_camX = np.matmul(np.linalg.inv(origin_T_camRs[b,s]),
                                    origin_T_camXs[b,s])
            xyz_camR = utils_py.apply_4x4(camR_T_camX, xyz_cam)

            save_pngs_to_disk = False
            if (save_pngs_to_disk):
                MH = 256
                MW = 256
                MD = 256
                proto = np.zeros([MH, MW, MD])
                occX = utils_py.voxelize_xyz(xyz_cam, proto)
                occR = utils_py.voxelize_xyz(xyz_camR, proto)
                occX_im = np.squeeze(np.max(occX, axis=0))
                occR_im = np.squeeze(np.max(occR, axis=0))
                imageio.imwrite('%05d_occX_cam%02d.png' % (sample_inds[b], s), occX_im)
                imageio.imwrite('%05d_occR_cam%02d.png' % (sample_inds[b], s), occR_im)

    # ok we are done
    # let's get to float32 and feed to tf

    # put rgbs in [-.5, .5]
    rgbs = rgbs - 0.5

    rgb_camXs = rgbs.astype(np.uint8)
    depth_camXs = depths.astype(np.float32)
    xyz_camXs = xyz_cams.astype(np.float32)
    pix_T_cams = pix_T_cams.astype(np.float32)
    origin_T_camRs = origin_T_camRs.astype(np.float32)
    origin_T_camXs = origin_T_camXs.astype(np.float32)

    d = dict()
    d['pix_T_cams'] = pix_T_cams
    d['rgb_camXs'] = rgb_camXs
    d['xyz_camXs'] = xyz_camXs
    d['depth_camXs'] = depth_camXs
    d['origin_T_camRs'] = origin_T_camRs
    d['origin_T_camXs'] = origin_T_camXs
    d['puck_xyz_camRs'] = puck_xyz_camRs
    d['camRs_T_puck'] = camRs_T_puck


    # info shared across seqs
    basic_info = dict()
    basic_info["T"] = B
    basic_info["S"] = S
    basic_info["image_width"] = hyp.W
    basic_info["image_height"] = hyp.H

    return d, basic_info

# Convert the given data to records
def convert_to_tfrecords(data, data_dict, tfrecord_filename):
    tfrecord_options = tf.io.TFRecordOptions('GZIP')
    # length of record written to each file
    record_files = []
    n = data_dict["rgb_camXs"].shape[0]
    for i, t in enumerate(range(0, n)):
        # import pdb; pdb.set_trace()
        example = tf.train.Example(features=tf.train.Features(feature={
            'rgb_camXs': utils_py.bytes_feature(data_dict['rgb_camXs'][i].tostring()), #uint8
            'depth_camXs': utils_py.bytes_feature(data_dict['depth_camXs'][i].tostring()),
            'pix_T_cams': utils_py.bytes_feature(data_dict['pix_T_cams'][i].tostring()),
            'origin_T_camXs': utils_py.bytes_feature(data_dict['origin_T_camXs'][i].tostring()),
            #'origin_T_camRs': utils_py.bytes_feature(data_dict['origin_T_camRs'][i].tostring()),
            #'puck_xyz_camRs': utils_py.bytes_feature(data_dict['puck_xyz_camRs'][i].tostring()),
            #'camRs_T_puck': utils_py.bytes_feature(data_dict['camRs_T_puck'][i].tostring()),
            #'observation': utils_py.bytes_feature(np.hstack([data['desired_goal'][i], data['achieved_goal'][i]]).tostring())
            }))
        file = tfrecord_filename.split('/')[-1] + "_" + str(i) + ".tf_records"
        record_files.append(file + "\n")
        with tf.io.TFRecordWriter(tfrecord_filename + "_" + str(i) + ".tf_records", options=tfrecord_options) as writer:
            writer.write(example.SerializeToString())
    return record_files

