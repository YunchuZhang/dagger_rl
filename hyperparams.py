import os
import getpass

# Updated the values 
B = 8 #4 # batch size
H = 64 #240 # height
W = 64 #320 # width
####################

BY = 200*2 # bird height (y axis, [-40, 40])
BX = 176*2 # bird width (x axis, [0, 70.4])
BZ = 20 # bird depth (z axis, [-3.0, 1.0])

MH = 200*2
MW = 176*2
MD = 20

PH = int(128/4)
PW = int(384/4)

ZY = 32 
ZX = 32 
ZZ = 16 

N = 50 # number of boxes produced by the rcnn (not all are good)
K = 1 # number of boxes to actually use
S = 2 # seq length
T = 256 # height & width of birdview map
V = 100000 # num velodyne points

#----------- loading -----------#
emb_init = ""
feat_init = ""
obj_init = ""
box_init = ""
ort_init = ""
inp_init = ""
traj_init = ""
occ_init = ""
view_init = ""
vis_init = ""
flow_init = ""
ego_init = ""
total_init = ""
reset_iter = False

do_freeze_emb = False
do_freeze_feat = False
do_freeze_obj = False
do_freeze_box = False
do_freeze_ort = False
do_freeze_inp = False
do_freeze_traj = False
do_freeze_occ = False
do_freeze_view = False
do_freeze_vis = False
do_freeze_flow = False
do_freeze_ego = False
do_resume = False
do_profile = False


# eval mode: save npys
do_eval_map = False
do_eval_recall = False # keep a buffer and eval recall within it
do_save_embs = False
do_save_ego = False

#----------- augs -----------#
# do_aug2D = False
# do_aug3D = False
do_aug_color = False
do_horz_flip = False
do_synth_rt = False
do_synth_nomotion = False
do_piecewise_rt = False
do_sparsify_pointcloud = 0 # choose a number here, for # pts to use

#----------- net design -----------#
# run nothing
do_emb = False
do_feat = False
do_obj = False
do_box = False
do_ort = False
do_inp = False
do_traj = False
do_occ = False
do_view = False
do_flow = False
do_ego = False
do_vis = False

#----------- general hypers -----------#
lr = 0.0

#----------- emb hypers -----------#
# emb3D = False
emb_2D3D_coeff = 0.0
emb_smooth_coeff = 0.0
emb_smooth3D_coeff = 0.0
emb_dim = 8
emb_use_aug = False
emb_2D_coeff = 0.0
emb_2D_l1_coeff = 0.0
emb_3D_coeff = 0.0
emb_3D_l1_coeff = 0.0
emb_3Dx_coeff = 0.0
emb_3Dx_l1_coeff = 0.0
emb_3Dm_coeff = 0.0
emb_3Dm_l1_coeff = 0.0
# emb_edgeloss_coeff = 1e-1
emb_samp_sz        = 4*6
emb_samp           = 'gridtex' # tex,rand,gridrand,gridtex
emb_do_subsamp     = True
emb_grid_cell_sz   = [30,40]
# emb_loss           = 'lifted'

#----------- feat hypers -----------#
feat_coeff = 0.0
feat_rigid_coeff = 0.0
feat_do_vae = False
feat_kl_coeff = 0.0
feat_dim = 8

#----------- obj hypers -----------#
obj_coeff = 0.0
obj_dim = 8

#----------- box hypers -----------#
box_sup_coeff = 0.0
box_cs_coeff = 0.0
box_dim = 8

#----------- ort hypers -----------#
ort_coeff = 0.0
ort_warp_coeff = 0.0
ort_dim = 8

#----------- inp hypers -----------#
inp_coeff = 0.0
inp_dim = 8

#----------- traj hypers -----------#
traj_coeff = 0.0
traj_dim = 8

#----------- occ hypers -----------#
occ_do_cheap = False
occ_coeff = 0.0
occ_smooth_coeff = 0.0

#----------- view hypers -----------#
view_depth = 64
view_pred_embs = False
view_pred_rgb = False
view_use_halftanh = False
view_l1_coeff = 0.0
view_ce_coeff = 0.0
view_dl_coeff = 0.0

#----------- vis hypers-------------#
vis_softmax_coeff = 0.0
vis_hard_coeff = 0.0
vis_l1_coeff = 0.0
vis_debug = False

#----------- flow hypers -----------#
flow_use_sig = False
flow_sig_coeff = 0.0
flow_warp_coeff = 0.0
flow_rgb_coeff = 0.0
flow_smooth_coeff = 0.0
flow_huber_coeff = 0.0
flow_synth_huber_coeff = 0.0

#----------- ego hypers -----------#
ego_use_gt = False
ego_use_precomputed = False
ego_rtd_coeff = 0.0
ego_rta_coeff = 0.0
ego_traj_coeff = 0.0
ego_warp_coeff = 0.0

#----------- mod -----------#

mod = '""'

############ slower-to-change hyperparams below here ############

## logging
log_freq_train = 100
log_freq_val = 100
log_freq_test = 100
snap_freq = 2000

max_iters = 10000
shuffle_train = True
shuffle_val = True
shuffle_test = True

dataset_name = ""
seqname = ""

trainset = ""
valset = ""
testset = ""

dataset_list_dir = ""
dataset_location = ""

# mode selection
do_zoom = False
do_carla_det = False
do_carla_mot = False
do_carla_flo = False
do_carla_sta = False
do_mujoco_offline = False

############ rev up the experiment ############
############ make some final adjustments ############

if not do_mujoco_offline:
    trainset_path = "%s/%s.txt" % (dataset_list_dir, trainset)
    valset_path = "%s/%s.txt" % (dataset_list_dir, valset)
    testset_path = "%s/%s.txt" % (dataset_list_dir, testset)
else:
    trainset_path = "%s/%s.npy" % (dataset_location, trainset)
    valset_path = "%s/%s.npy" % (dataset_location, valset)
    testset_path = "%s/%s.npy" % (dataset_location, testset)
    
data_paths = {}
data_paths['train'] = trainset_path
data_paths['val'] = valset_path
data_paths['test'] = testset_path

set_nums = {}
set_nums['train'] = 0
set_nums['val'] = 1
set_nums['test'] = 2

set_names = ['train', 'val', 'test']

log_freqs = {}
log_freqs['train'] = log_freq_train
log_freqs['val'] = log_freq_val
log_freqs['test'] = log_freq_test

shuffles = {}
shuffles['train'] = shuffle_train
shuffles['val'] = shuffle_val
shuffles['test'] = shuffle_test


############ autogen a name; don't touch any hypers! ############

def strnum(x):
    s = '%g' % x
    if '.' in s:
        s = s[s.index('.'):]
    return s

# name = "%02dx%dx%d_%.1e" % (B, H, W, lr)
# name = "%02d" % B
# name = "%02dx%dx%dx%d" % (B, BY, BX, BZ)
# name = "%02d_b%dx%dx%d_z%dx%dx%d_p%dx%d_k%d" % (B, BY,BX,BZ, ZY,ZX,ZZ, PH,PW, K)

# if do_cups_static or do_carla_static:
#     name = "%02d_b%dx%dx%d_p%dx%d" % (B, BY,BX,BZ, PH,PW)
# elif do_carla_surveil or do_carla_moving or do_carla_det:
#     # name = "%02d_b%dx%dx%d_p%dx%d_k%d" % (B, BY,BX,BZ, PH,PW, K)
#     name = "%02d_b%dx%dx%d_z%dx%dx%d_p%dx%d_%d" % (B, BY,BX,BZ, ZY,ZX,ZZ, PH,PW, K)
# elif do_cups_moving:
#     name = "%02d_b%dx%dx%d_p%dx%d" % (B, BY,BX,BZ, PH,PW)
# elif do_carla_oneobj:
#     # name = "%02d_b%dx%dx%d_z%dx%dx%d_p%dx%d" % (B, BY,BX,BZ, ZY,ZX,ZZ, PH,PW)
#     name = "%02d_b%dx%dx%d_z%dx%dx%d_p%dx%d_%d" % (B, BY,BX,BZ, ZY,ZX,ZZ, PH,PW, K)
# else:
#     assert(False) # please choose a model name starter

# name = "%02d_m%dx%dx%d_z%dx%dx%d_p%dx%d_k%d" % (B, MH,MW,MD, ZH,ZW,ZD, PH,PW, K)
# name = "%02d_m%dx%dx%d_p%dx%d_k%d" % (B, MH,MW,MD, PH,PW, K)
name = "%02d_m%dx%dx%d_p%dx%d" % (B, MH,MW,MD, PH,PW)


if lr > 0.0:
    lrn = "%.1e" % lr
    # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]
    name += "_%s" % lrn

if do_feat:
    name += "_F"
    name += "%d" % feat_dim
    if feat_do_vae:
        name += "v"

    if do_freeze_feat:
        name += "f"
    else:
        feat_losses = [feat_rigid_coeff,
                       feat_kl_coeff,
        ]
        feat_prefixes = ["r",
                         "k",
        ]
        for l_, l in enumerate(feat_losses):
            if l > 0:
                name += "_%s%s" % (feat_prefixes[l_],strnum(l))
        

if do_ego:
    name += "_G"
    if ego_use_gt:
        name += "gt"
    elif ego_use_precomputed:
        name += "pr"
    else:
        if do_freeze_ego:
            name += "f"
        else:
            ego_losses = [ego_rtd_coeff,
                          ego_rta_coeff,
                          ego_traj_coeff,
                          ego_warp_coeff,
            ]
            ego_prefixes = ["rtd",
                            "rta",
                            "t",
                            "w",
            ]
            for l_, l in enumerate(ego_losses):
                if l > 0:
                    name += "_%s%s" % (ego_prefixes[l_],strnum(l))
    
if do_obj:
    name += "_J"
    # name += "%d" % obj_dim

    if do_freeze_obj:
        name += "f"
    else:
        # no real hyps here
        pass
        
if do_box:
    name += "_B"
    # name += "%d" % box_dim

    if do_freeze_box:
        name += "f"
    else:
        box_coeffs = [box_sup_coeff,
                      box_cs_coeff,
                      # box_smooth_coeff,
        ]
        box_prefixes = ["su",
                        "cs",
                        # "s",
        ]
        for l_, l in enumerate(box_coeffs):
            if l > 0:
                name += "_%s%s" % (box_prefixes[l_],strnum(l))
        
        
if do_ort:
    name += "_O"
    # name += "%d" % ort_dim

    if do_freeze_ort:
        name += "f"
    else:
        ort_coeffs = [ort_coeff,
                      ort_warp_coeff,
                      # ort_smooth_coeff,
        ]
        ort_prefixes = ["c",
                        "w",
                        # "s",
        ]
        for l_, l in enumerate(ort_coeffs):
            if l > 0:
                name += "_%s%s" % (ort_prefixes[l_],strnum(l))
        
if do_inp:
    name += "_I"
    # name += "%d" % inp_dim

    if do_freeze_inp:
        name += "f"
    else:
        inp_coeffs = [inp_coeff,
                      # inp_smooth_coeff,
        ]
        inp_prefixes = ["c",
                        # "s",
        ]
        for l_, l in enumerate(inp_coeffs):
            if l > 0:
                name += "_%s%s" % (inp_prefixes[l_],strnum(l))
        
if do_traj:
    name += "_T"
    name += "%d" % traj_dim

    if do_freeze_traj:
        name += "f"
    else:
        # no real hyps here
        pass
        
if do_occ:
    name += "_O"
    if occ_do_cheap:
        name += "c"
    if do_freeze_occ:
        name += "f"
    else:
        occ_coeffs = [occ_coeff,
                      occ_smooth_coeff,
        ]
        occ_prefixes = ["c",
                        "s",
        ]
        for l_, l in enumerate(occ_coeffs):
            if l > 0:
                name += "_%s%s" % (occ_prefixes[l_],strnum(l))
        
if do_view:
    name += "_V"
    if view_pred_embs:
        name += "e"
    if view_pred_rgb:
        name += "r"
    if view_use_halftanh:
        name += "h"
    if do_freeze_view:
        name += "f"
    
    # sometimes, even if view is frozen, we use the loss
    # to train other nets
    view_coeffs = [view_depth,
                   view_l1_coeff,
                   view_ce_coeff,
                   view_dl_coeff,
    ]
    view_prefixes = ["d",
                     "c",
                     "e",
                     "s",
    ]
    for l_, l in enumerate(view_coeffs):
        if l > 0:
            name += "_%s%s" % (view_prefixes[l_],strnum(l))

if do_vis:
    name += "_V"
    if vis_debug:
        name += 'd'
    if do_freeze_vis:
        name += "f"
    else:
        vis_coeffs = [vis_softmax_coeff,
                      vis_hard_coeff,
                      vis_l1_coeff,
        ]
        vis_prefixes = ["s",
                        "h",
                        "c",
        ]
        for l_, l in enumerate(vis_coeffs):
            if l > 0:
                name += "_%s%s" % (vis_prefixes[l_],strnum(l))
            
        
if do_emb:
    name += "_E"
    name += "%d" % emb_dim
    # if emb3D:
    #     name += '3'
    if emb_use_aug:
        name += "a"
    if do_freeze_emb:
        name += "f"
    # else:
    
    # sometimes, even if emb is frozen, we use the loss
    # to train other nets
    
    emb_coeffs = [emb_2D3D_coeff,
                  emb_smooth_coeff,
                  emb_smooth3D_coeff,
                  emb_2D_coeff,
                  emb_2D_l1_coeff,
                  emb_3D_coeff,
                  emb_3D_l1_coeff,
                  emb_3Dx_coeff,
                  emb_3Dx_l1_coeff,
                  emb_3Dm_coeff,
                  emb_3Dm_l1_coeff,
    ]
    emb_prefixes = ["y",
                    "s",
                    "v",
                    "a",
                    "b",
                    "i",
                    "j",
                    "x",
                    "y",
                    "m",
                    "n",
    ]
    for l_, l in enumerate(emb_coeffs):
        if l > 0:
            name += "_%s%s" % (emb_prefixes[l_],strnum(l))
                
if do_flow:
    name += "_F"
    if flow_use_sig:
        name += "s"
    if do_freeze_flow:
        name += "f"
    else:
        flow_coeffs = [flow_warp_coeff,
                       flow_rgb_coeff,
                       flow_smooth_coeff,
                       flow_huber_coeff,
                       flow_synth_huber_coeff,
                       flow_sig_coeff,
        ]
        flow_prefixes = ["w",
                         "r",
                         "s",
                         "h",
                         "y",
                         "p",
        ]
        for l_, l in enumerate(flow_coeffs):
            if l > 0:
                name += "_%s%s" % (flow_prefixes[l_],strnum(l))
        
##### end model description

# add some training data info

sets_to_run = {}
if trainset:
    name = "%s_%s" % (name, trainset)
    sets_to_run['train'] = True
else:
    sets_to_run['train'] = False
    
if valset:
    name = "%s_%s" % (name, valset)
    sets_to_run['val'] = True
else:
    sets_to_run['val'] = False
    
if testset:
    name = "%s_%s" % (name, testset)
    sets_to_run['test'] = True
else:
    sets_to_run['test'] = False

sets_to_train = {}
sets_to_train['train'] = True
sets_to_train['val'] = False
sets_to_train['test'] = False

if (do_aug_color or
    do_horz_flip or
    do_synth_rt or
    do_piecewise_rt or
    do_synth_nomotion or
    do_sparsify_pointcloud):
    name += "_A"
    if do_aug_color:
        name += "c"
    if do_horz_flip:
        name += "h"
    if do_synth_rt:
        assert(not do_piecewise_rt)
        name += "s"
    if do_piecewise_rt:
        assert(not do_synth_rt)
        name += "p"
    if do_synth_nomotion:
        name += "n"
    if do_sparsify_pointcloud:
        name += "v"
    
if (not shuffle_train) or (not shuffle_val) or (not shuffle_test):
    name += "_ns"


if do_profile:
    name += "_PR"

if mod:
    name = "%s_%s" % (name, mod)
    
if do_resume:
    total_init = name

print(name)
