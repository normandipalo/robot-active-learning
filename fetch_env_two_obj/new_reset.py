import numpy as np

def reset_sim(env):
        self = env.unwrapped
        env.sim.set_state(self.initial_state)
       # self.goal = np.clip(np.random.randn(3)/3, 0, 0.5)
        
       # self.goal[2] = 0.425
        # Randomize start position of object.
        if self.has_object:
            object_xpos0 = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos0 - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos0 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos0 = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos0.shape == (7,)
            object_qpos0[:2] = object_xpos0
            self.sim.data.set_joint_qpos('object0:joint', object_qpos0)
            
            object_xpos1 = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos1 - self.initial_gripper_xpos[:2]) < 0.1 or np.linalg.norm(object_xpos1 - object_xpos0) < 0.1 \
            or np.linalg.norm(object_xpos1 - self.goal[:2]) < 0.2:
                object_xpos1 = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos1 = self.sim.data.get_joint_qpos('object1:joint')
            assert object_qpos1.shape == (7,)
            object_qpos1[:2] = object_xpos1
            self.sim.data.set_joint_qpos('object1:joint', object_qpos1)

        self.sim.forward()
        return True
