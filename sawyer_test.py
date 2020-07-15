from metaworld.envs.mujoco.sawyer_xyz import *
import numpy as np

#check_bits = np.zeros(4)

"""
def check(env, obs, check_bits):
 # check_bits = np.zeros(4)
  if np.linalg.norm(obs[:2] - obs[3:5]) < 0.005: check_bits[0] = 1 #above it
  if np.linalg.norm(obs[2] - obs[5]) < 0.03: check_bits[1] = 1 #same height (graspable if [0] == 1)
  if obs[5] > 0.03: check_bits[2] = 1 #above ground (probably in hand)
  if np.linalg.norm(obs[:2] - np.array([0.075,0.01]) - obs[6:8]) < 0.0105: check_bits[3] = 1
  goal = obs[6:9]
  goal[2] = 0.
  print("check bits", check_bits)
  return check_bits
"""  
  
# I try a different version in which a "check" puts to 1 every previous check. Could fix a bug.  
def check(env, obs, check_bits):
 # check_bits = np.zeros(4)
  if np.linalg.norm(obs[:2] - obs[3:5]) < 0.005: check_bits[0] = 1 #above it
  if np.linalg.norm(obs[2] - obs[5]) < 0.03: 
    check_bits[0] = 1
    check_bits[1] = 1 #same height (graspable if [0] == 1)
  if obs[5] > 0.03: 
    check_bits[0] = 1
    check_bits[1] = 1
    check_bits[2] = 1 #above ground (probably in hand)
  if np.linalg.norm(obs[:2] - np.array([0.075,0.01]) - obs[6:8]) < 0.0105: 
    check_bits[0] = 1
    check_bits[1] = 1
    check_bits[2] = 1
    check_bits[3] = 1
  goal = obs[6:9]
  goal[2] = 0.
 # print("check bits", check_bits)
  return check_bits

def check_success(env, obs):
    goal = obs[6:9]
    goal[2] = 0.
  #  print("1",np.linalg.norm( - obs[3:5] + np.array([0.077,0.]) + goal[:2]))
  #  print("2",obs[5], "\n")
    if np.linalg.norm( - obs[3:5] + np.array([0.077,0.]) + goal[:2]) < 0.04 and obs[5] < 0.088:
        return True
    return False


def get_act(env, obs, cb, norm = False):
#    cb = check(env, obs)
    act = np.array([0.,0.,0.,0.])
    act[:3] = obs[3:6] - obs[:3]
    if cb[0] == 0:
        if obs[2] < 0.18:
            act[2] = 0.3
            act[3] = -1.
        else:
            act[2] = 0.
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.3
        steps = 1
    elif cb[1] == 0:
        act[:2] = 0
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.3
        steps = 1
    elif cb[2] == 0:
        act = np.array([0.,0.,0.,0])
        cb[2] = 1
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.3
        act[3] =1
        steps = 25
    elif cb[3] == 0:
        act = np.array([0.,0.,0.,0])
        act[:3] = obs[6:9] - obs[:3] + np.array([0.075,0.,0.])
        act[2] = 0.2 if obs[2] < 0.3 else 0.
        act/=(np.linalg.norm(act) + 0.01)
        act*=0.3
        act[3] = 1
        steps = 1
    else:
        act = np.array([0.,0.,-0.2,0.])
        act/=np.linalg.norm(act)
        act[3] = 1
        act*=0.3
        steps = 1
    #    act[:3]+=np.random.randn(3)*0.1

    if cb[2] == 1:
        act[-1] = 1.
    else:
        act[-1] = 0.
    return act, steps


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
