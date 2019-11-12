from metaworld.envs.mujoco.sawyer_xyz import *
import numpy as np

check_bits = np.zeros(4)

def check(env, obs):
  if np.linalg.norm(obs[:2] - obs[3:5]) < 0.001: check_bits[0] = 1 #above it
  if np.linalg.norm(obs[2] - obs[5]) < 0.03: check_bits[1] = 1 #same height (graspable if [0] == 1)
  if obs[5] > 0.02: check_bits[2] = 1
  if np.linalg.norm(obs[:2] - np.array([0.075,0.01]) - obs[6:8]) < 0.0215: check_bits[3] = 1
  print(np.linalg.norm(obs[3:5] - np.array([0.075,0.]) - obs[6:8]))
  return check_bits

def check_success(env, obs):
    if np.linalg.norm(obs[:3] + np.array([0.075,0.,0.]) - obs[6:9]) < 0.16:
        print("SUCCESS")

def get_act(env, obs):
    cb = check(env, obs)
    act = obs[3:6] - obs[:3]
    if cb[0] == 0:
        act[2] = 0
        return act, 1
    elif cb[1] == 0:
        act[:2] = 0
        return act, 1
    elif cb[2] == 0:
        act = np.array([0.,0.,0.,1])
        cb[2] = 1
        return act, 15
    elif cb[3] == 0:
        act = np.array([0.,0.,0.,1])
        act[:3] = obs[6:9] - obs[:3] + np.array([0.075,0.,0.])
        act[2] = 0.5 if obs[2] < 0.3 else 0.
        return act, 1
    else:
        act = np.array([0.,0.,-0.2,1])
        return act, 10







env = SawyerNutAssemblyEnv()
obs = env.reset()
env.render()
cb = check(env, obs)

while 1:
  act, steps = get_act(env, obs)
  for i in range(steps):
      obs, *_ = env.step(act)
      env.render()
      check_success(env, obs)
      cb = check(env, obs)
