import gym
import numpy as np

class CameraRobot():
    def __init__(self, env):
        self.env = env
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
        return (state, s, d), 0, False, ""

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
        return (state, s, d)

    def seed(self, seed):
        self.env.seed(seed)
