import gym
import numpy as np
import cv2
from gym.envs.robotics import rotations, robot_env, utils


class CameraRobot():
    def __init__(self, env, im_dim):
        self.env = env
        self.im_dim = im_dim
    #    self.env.env.render(mode = "rgb_array")

    def step(self, act):
        state, *_ = self.env.step(act)
        self.env.render(mode = "rgb_array")
        #Have to call two times to make it work.
        s = self.env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
        a, d = self.env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')
        s = self.env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
        a, d = self.env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')
        d = d[::-1,:].astype(np.float64)[:,:,None]
        #return (state, cv2.resize(s[16:44,:,:], (self.im_dim, self.im_dim)), cv2.resize(d[16:44,:,:], (self.im_dim,self.im_dim))), \
        return (state, cv2.resize(s, (self.im_dim, self.im_dim)), cv2.resize(d, (self.im_dim,self.im_dim))), \
                0, False, ""

    def reset(self):
        state = self.env.reset()
        self.env.render(mode = "rgb_array")
        #Have to call two times to make it work.
        s = self.env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
        a, d = self.env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')
        s = self.env.env.viewer.sim.render(50, 50, camera_name = 'external_camera_0')[::-1,:,:].astype(np.float64)/255.
        a, d = self.env.env.sim.render(50, 50, depth = True, camera_name = 'external_camera_0')


        d = d[::-1,:].astype(np.float64)[:,:,None]
#        self.env.render()
        #return (state, cv2.resize(s[16:44,:,:], (self.im_dim, self.im_dim)), cv2.resize(d[16:44,:,:], (self.im_dim, self.im_dim)))
        return (state, cv2.resize(s, (self.im_dim, self.im_dim)), cv2.resize(d, (self.im_dim, self.im_dim)))

    def seed(self, seed):
        self.env.seed(seed)

class Fetch2Cubes():
    def __init__(self, env):
        self.env = env

    def render(self, mode = "human"):
        self.env.render(mode)

    def reset(self):
        self.env.reset()
        self.reset_sim(self.env)
        return self._get_obs()

    def step(self, act):
        next_state, a, b, c = self.env.step(act)
        next_state = self._get_obs()
        return next_state, a, b, c

    def reset_sim(self, env):
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

    def _get_obs(self):
        self = self.env
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos0 = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot0 = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp0 = self.sim.data.get_site_xvelp('object0') * dt
            object_velr0 = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos0 = object_pos0 - grip_pos
            object_velp0 -= grip_velp

            object_pos1 = self.sim.data.get_site_xpos('object1')
            # rotations
            object_rot1 = rotations.mat2euler(self.sim.data.get_site_xmat('object1'))
            # velocities
            object_velp1 = self.sim.data.get_site_xvelp('object1') * dt
            object_velr1 = self.sim.data.get_site_xvelr('object1') * dt
            # gripper state
            object_rel_pos1 = object_pos1 - grip_pos
            object_velp1 -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.concatenate((np.squeeze(object_pos0.copy()),np.squeeze(object_pos1.copy()) ))
        obs = np.concatenate([
            grip_pos, object_pos0.ravel(), object_rel_pos0.ravel(), object_pos1.ravel(), object_rel_pos1.ravel(),
             gripper_state, object_rot0.ravel(),
            object_velp0.ravel(), object_velr0.ravel(), object_rot1.ravel(),
            object_velp1.ravel(), object_velr1.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def seed(self, seed):
        self.env.seed(seed)
