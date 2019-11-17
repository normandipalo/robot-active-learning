from metaworld.envs.mujoco.sawyer_xyz import *
import numpy as np

#check_bits = np.zeros(4)

def check(env, obs, check_bits):
 # check_bits = np.zeros(4)
  if np.linalg.norm(obs[:2] - obs[3:5]) < 0.005: check_bits[0] = 1 #above it
  if np.linalg.norm(obs[2] - obs[5]) < 0.03: check_bits[1] = 1 #same height (graspable if [0] == 1)
  if obs[5] > 0.02: check_bits[2] = 1
  if np.linalg.norm(obs[:2] - np.array([0.075,0.01]) - obs[6:8]) < 0.0215: check_bits[3] = 1
  goal = obs[6:9]
  goal[2] = 0.
  return check_bits

def check_success(env, obs):
    goal = obs[6:9]
    goal[2] = 0.
    if np.linalg.norm( - obs[3:5] + np.array([0.077,0.]) + goal[:2]) < 0.03 and obs[5] < 0.088:
        return True
    return False


def get_act(env, obs, cb, norm = False):
#    cb = check(env, obs)
    act = np.array([0.,0.,0.,0.])
    act[:3] = obs[3:6] - obs[:3]
    if cb[0] == 0:
        act[2] = 0
        act/=(np.linalg.norm(act) + 0.01)
        return act*0.2, 1
    elif cb[1] == 0:
        act[:2] = 0
        act/=(np.linalg.norm(act) + 0.01)
        return act*0.2, 1
    elif cb[2] == 0:
        act = np.array([0.,0.,0.,0])
        cb[2] = 1
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.2
        act[3] =1
        return act, 15
    elif cb[3] == 0:
        act = np.array([0.,0.,0.,0])
        act[:3] = obs[6:9] - obs[:3] + np.array([0.075,0.,0.])
        act[2] = 0.2 if obs[2] < 0.3 else 0.
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.2
        act[3] = 1
        return act, 1
    else:
        act = np.array([0.,0.,-0.2,0.])
        act/=np.linalg.norm(act)
        act[3] = 1
        return act*0.2, 10


if __name__ == "__main__":
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
