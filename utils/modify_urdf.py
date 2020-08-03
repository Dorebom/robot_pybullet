from lxml import etree

class UrdfDesigner():
    def __init__(self):
        pass

    def load(self,name='robot'):
        if name == 'robot':
            urdf_file = 'urdf/kuka_iiwa_6/basic_model.urdf'
            with open(urdf_file) as f:
                self.tree = etree.parse(f)
                self.name = name

            #self._set_joint_type_info("lbr_iiwa_joint_3", "fixed")

    def export(self):
        if self.name == 'robot':
            self.tree.write(
                'urdf/kuka_iiwa_6/modified_model.urdf',
                pretty_print = True,
                xml_declaration = True,
                encoding = "utf-8" )

    def _get_link_size_info(self, link_name, attr_name = 'visual'):
        return self.tree.xpath("/robot/link[@name='"+link_name+"']/"+attr_name+"/geometry/box")

    def _get_link_pose_info(self, link_name, attr_name = 'visual'):
        return self.tree.xpath("/robot/link[@name='"+link_name+"']/"+attr_name+"/origin")

    def _get_joint_pose_info(self, joint_name):
        return self.tree.xpath("/robot/joint[@name='"+joint_name+"']/origin")

    def _set_size_info(self, states, size):
        if len(states):
            for state in states:
                state.attrib['size'] = str(size[0]) + ' ' + str(size[1]) + ' ' + str(size[2])

    def _set_pose_info(self, states, pose):
        if len(states):
            for state in states:
                state.attrib['xyz'] = str(pose[0]) + ' ' + str(pose[1]) + ' ' + str(pose[2])
                state.attrib['rpy'] = str(pose[3]) + ' ' + str(pose[4]) + ' ' + str(pose[5])

    def _set_joint_type_info(self, name, type):
        states = self.tree.xpath("/robot/joint[@name='"+name+"']")
        for state in states:
            state.attrib['type'] = type
            print(state)

    def modify_link_size(self, size, name='lbr_iiwa_link_0'):
        attr_name = 'visual'
        states = self._get_link_size_info(name, attr_name)
        self._set_size_info(states, size)
        attr_name = 'collision'
        states = self._get_link_size_info(name, attr_name)
        self._set_size_info(states, size)

    def modify_link_pose(self, pose, name='lbr_iiwa_link_0'):
        attr_name = 'visual'
        states = self._get_link_pose_info(name, attr_name)
        self._set_pose_info(states, pose)
        attr_name = 'collision'
        states = self._get_link_pose_info(name, attr_name)
        self._set_pose_info(states, pose)

    def modify_joint_pose(self, pose, name='lbr_iiwa_joint_1'):
        states = self._get_joint_pose_info(name)
        self._set_pose_info(states, pose)

    def modify_tcp_pose(self, pose):
        states = self._get_joint_pose_info('tcp_joint')
        self._set_pose_info(states, pose)

