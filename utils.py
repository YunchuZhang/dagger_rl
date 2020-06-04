import os
from xml.etree import ElementTree as et

def change_env_to_use_correct_mesh(mesh):
	path_to_xml = os.path.join('/home/apokle/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
	#path_to_xml = os.path.join('/Users/apokle/Documents/goal_conditioned_policy/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
	tree = et.parse(path_to_xml)
	root = tree.getroot()
	[x.attrib for x in root.iter('geom')][0]['mesh']=mesh
	#[x.attrib for x in root.iter('geom')][0]['size']=scale
	 #set the masses, inertia and friction in a plausible way

	physics_dict = {}
	physics_dict["printer"] =  ["6.0", ".00004 .00003 .00004", "1 1 .0001" ]
	physics_dict["mug1"] =  ["0.31", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
	physics_dict["mug2"] =  ["16.5", ".000001 .0000009 .0000017", "0.4 0.2 .00001" ]
	physics_dict["mug3"] =  ["0.33", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
	physics_dict["can1"] =  ["0.55", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["car1"] =  ["0.2", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car2"] =  ["0.4", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car3"] =  ["5.5", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car4"] =  ["0.8", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["car5"] =  ["2.0", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
	physics_dict["boat"] =  ["17.0", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["bowl"] =  ["10", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["bowl2"] =  ["1", ".00002 .00002 .00001", "0.2 0.2 .0001" ]
	physics_dict["bowl4"] =  ["0.7", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["hat1"] =  ["0.2", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["hat2"] =  ["0.4", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
	physics_dict["mouse"] = ["2.7", ".00027 .00025 .00016", "1.5 0.5 .000001"]
	physics_dict["book"] = ["10", ".00768 .01193 .00646", "3.5 2.5 .000001"]
	physics_dict["coffee_mug"] = ["21", ".0007 .0002 .0007", "0.35 0.25 .000001"]
	physics_dict["boat2"] =  ["6.0", ".00002 .00002 .00001", "0.2 0.2 .0001" ]
	physics_dict["headphones"] =  ["3", ".0012 .0039 .0029", "0.7 0.4 .0001" ]
	physics_dict["ball"] =  ["9", "0.000007 0.000007 0.000007", "0.0005 0.0004 .0001" ]
	physics_dict["eyeglass"] =  ["2.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["plane"] =  ["5.5", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["hamet"] =  ["12.5", "0.00016 0.00023 0.00008", "0.005 0.004 .001" ]
	physics_dict["clock"] =  ["3.5", "0.00016 0.00023 0.00008", "0.00005 0.00004 .00001" ]
	physics_dict["skate"] =  ["12", "0.00016 0.00023 0.00008", "0.6 0.4 .0001" ]
	physics_dict["bag1"] =  ["3", "0.00016 0.00023 0.00008", "0.005 0.004 .0001" ]
	physics_dict["bag2"] =  ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001" ]
	physics_dict["keyboard"] =  ["3", "0.00016 0.00023 0.00008", "0.002 0.004 .0001" ]
	physics_dict["knife"] =  ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]
	physics_dict["pillow"] =  ["6", "0.00016 0.00023 0.00008", "0.5 0.4 .0001" ]
	physics_dict["bag22"] =  ["8", "0.00016 0.00023 0.00008", "0.01 0.01 .0001" ]
	physics_dict["knife2"] =  ["8", "0.00016 0.00023 0.00008", "0.0005 0.0004 .0001" ]

	#set parameters
	[x.attrib for x in root.iter('geom')][0]['mass'] = physics_dict[mesh][0]
	# [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
	[x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

	tree.write(path_to_xml)

def change_env_to_rescale_mesh(mesh, scale=1.0):
	path_to_xml = os.path.join('/home/apokle/multiworld/multiworld/envs/assets/sawyer_xyz/shared_config.xml')
	#path_to_xml = os.path.join('/Users/apokle/Documents/goal_conditioned_policy/multiworld/multiworld/envs/assets/sawyer_xyz/shared_config.xml')
	tree = et.parse(path_to_xml)
	root = tree.getroot()
	# import pdb; pdb.set_trace()
	for x in root.iter('mesh'):
		if x.attrib['name'] == mesh:
			x.attrib['scale'] = "{} {} {}".format(scale, scale, scale)
	tree.write(path_to_xml)