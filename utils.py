import os
import gym
# import multiworld
from glob import glob
from xml.etree import ElementTree as et
from quantized_policies.pcp_utils.utils import config_from_yaml_file, get_gym_dir, get_root_dir
from quantized_policies.pcp_utils.mesh_object import MeshObject
from quantized_policies.pcp_utils.parse_task_files import generate_integrated_xml

def change_env_to_use_correct_mesh(mesh):
    multiworld_path = os.path.dirname(multiworld.__file__)
    path_to_xml = os.path.join(multiworld_path, 'envs/assets/sawyer_xyz/sawyer_push_box.xml')
    tree = et.parse(path_to_xml)
    root = tree.getroot()
    [x.attrib for x in root.iter('geom')][0]['mesh']=mesh
    # [x.attrib for x in root.iter('geom')][0]['size'] = scale
    # set the masses, inertia and friction in a plausible way

    physics_dict = {}
    physics_dict["printer"] = ["6.0", ".00004 .00003 .00004", "1 1 .0001"]
    physics_dict["mug1"] = ["0.31", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001"]
    physics_dict["mug2"] = ["16.5", ".000001 .0000009 .0000017", "0.4 0.2 .00001"]
    physics_dict["mug3"] = ["0.33", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001"]
    physics_dict["can1"] = ["0.55", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["car1"] = ["0.2", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001"]
    physics_dict["car2"] = ["0.4", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001"]
    physics_dict["car3"] = ["5.5", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001"]
    physics_dict["car4"] = ["0.8", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001"]
    physics_dict["car5"] = ["2.0", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001"]
    physics_dict["boat"] = ["17.0", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["bowl"] = ["10", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["bowl2"] = ["1", ".00002 .00002 .00001", "0.2 0.2 .0001"]
    physics_dict["bowl4"] = ["0.7", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["hat1"] = ["0.2", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["hat2"] = ["0.4", ".00000002 .00000002 .00000001", "0.2 0.2 .0001"]
    physics_dict["mouse"] = ["2.7", ".00027 .00025 .00016", "1.5 0.5 .000001"]
    physics_dict["book"] = ["10", ".00768 .01193 .00646", "3.5 2.5 .000001"]
    physics_dict["coffee_mug"] = ["21", ".0007 .0002 .0007", "0.35 0.25 .000001"]
    physics_dict["boat2"] = ["6.0", ".00002 .00002 .00001", "0.2 0.2 .0001"]
    physics_dict["headphones"] = ["3", ".0012 .0039 .0029", "0.7 0.4 .0001"]
    physics_dict["ball"] = ["9", "0.000007 0.000007 0.000007", "0.0005 0.0004 .0001"]
    physics_dict["eyeglass"] = ["2.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001"]
    physics_dict["plane"] = ["5.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001"]
    physics_dict["hamet"] = ["12.5", "0.00016 0.00023 0.00008", "0.005 0.004 .001"]
    physics_dict["clock"] = ["3.5", "0.00016 0.00023 0.00008", "0.00005 0.00004 .00001"]
    physics_dict["skate"] = ["12", "0.00016 0.00023 0.00008", "0.6 0.4 .0001"]
    physics_dict["bag1"] = ["3", "0.00016 0.00023 0.00008", "0.005 0.004 .0001"]
    physics_dict["bag2"] = ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001"]
    physics_dict["keyboard"] = ["3", "0.00016 0.00023 0.00008", "0.002 0.004 .0001"]
    physics_dict["knife"] = ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001"]
    physics_dict["pillow"] = ["6", "0.00016 0.00023 0.00008", "0.5 0.4 .0001"]
    physics_dict["bag22"] = ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001"]
    physics_dict["knife2"] = ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001"]

    # set parameters
    [x.attrib for x in root.iter('geom')][0]['mass'] = physics_dict[mesh][0]
    # [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
    [x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

    tree.write(path_to_xml)

def change_env_to_rescale_mesh(mesh, scale=1.0):
    multiworld_path = os.path.dirname(multiworld.__file__)
    path_to_xml = os.path.join(multiworld_path, 'envs/assets/sawyer_xyz/shared_config.xml')
    tree = et.parse(path_to_xml)
    root = tree.getroot()
    for x in root.iter('mesh'):
        if x.attrib['name'] == mesh:
            x.attrib['scale'] = "{} {} {}".format(scale, scale, scale)
    tree.write(path_to_xml)

def make_env(env_name, base_xml_path=None, obj_name=None, task_config_path=None, **kwargs):
    if base_xml_path is not None:
        # process xml path
        if not base_xml_path.startswith('/'):
            base_xml_path = os.path.join(get_gym_dir(), base_xml_path)

        if not task_config_path.startswith('/'):
            repo_dir = os.path.dirname(os.path.realpath(__file__))
            base_xml_path = os.path.join(repo_dir, base_xml_path)

        # generate modified xml
        objs_config = config_from_yaml_file(task_config_path)
        obj = MeshObject(objs_config['objs'][obj_name], obj_name)
        # import ipdb;ipdb.set_trace()
        # ele = objs_config['objs'][obj_name]
        # scale = ele['scale'] if 'scale' in ele else 1.0
        # euler = ele['euler_xyz'] if 'euler_xyz' in ele else None
        # target_size = ele['target_size'] if 'target_size' in ele else None

        xml_path = generate_integrated_xml(base_xml_path, obj.obj_xml_file,
                                scale=obj.scale, 
                                # euler=obj.euler,
                                target_size=None, obj_name=obj_name)
        # xml_path = generate_integrated_xml(base_xml_path,
        #                                    obj.obj_xml_file,
        #                                    scale=obj.scale)
        env = gym.make(env_name, xml_path=xml_path, **kwargs)
    else:
        change_env_to_use_correct_mesh(obj_name)
        env = gym.make(env_name, **kwargs)
    return env

def get_latest_checkpoint(dir):
    """ Get latest checkpoint in a given directory. """
    ckpts = glob(os.path.join(dir, 'save*'))
    if len(ckpts) == 0:
        raise ValueError('No checkpoint found in directory {}'.format(dir))
    max_ckpt_index = 0
    latest_path = None
    for ckpt_path in ckpts:
        ckpt_index = int(ckpt_path.split('/')[-1][4:])
        max_ckpt_index = max(ckpt_index, max_ckpt_index)
        if ckpt_index == max_ckpt_index:
            latest_path = ckpt_path
    return latest_path
