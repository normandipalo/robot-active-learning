import tensorflow as tf
import gym
import numpy as np

import model
from ae import AE
import man_controller
import utils

if not tf.__version__ == "2.0.0-alpha0":
    tf.enable_eager_execution()

INITIAL_TRAIN_EPS = 100

BC_LR = 1e-3
BC_HD = 128
BC_HL = 2
BC_BS = 64
BC_EPS = 150

AE_HD = 128
AE_HL = 2
AE_LR = 1e-3
AE_BS = 64
AE_EPS = 150

TEST_EPS = 200
ACTIVE_STEPS_RETRAIN = 10
ACTIVE_ERROR_THR = 1.5


def get_experience(eps, env):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        new_states, new_acts = man_controller.get_demo(env, state)
        states+=new_states
        actions+=new_acts
                
    return states, actions

def test(model, test_set, env, xm, xs, render = False):
    successes, failures = 0,0
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = utils.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        for i in range(200):
            action = model((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)

            new_state, *_ = env.step(action[0])
         #   print(action)
            if render: env.render()
            state = new_state
       
            if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.07:
        #        print("SUCCESS!")
                succeded = 1
                successes +=1

                break
        if not succeded: 
       #     print("FAILURE")
            failures+=1
    return successes, failures

def get_active_exp(env, threshold, ae, xm, xs, render):
    
    err_avg = 0
    for i in range(20):
        state = env.reset()
        error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
        err_avg+=error
    err_avg/=20
    
    state = env.reset()
    error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
    #print("predicted error", error)

    tried = 0
    while not error > threshold*err_avg:
        tried+=1
        state = env.reset()
        error = ae.error((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)
  #      print("predicted error", error.numpy(), err_avg.numpy())
 #   print("Tried ", tried, " initial states")
    new_states, new_acts = man_controller.get_demo(env, state, render)

    return new_states, new_acts


def go(seed):
    if not tf.__version__ == "2.0.0-alpha0":
        tf.random.set_random_seed(seed)
    else: 
        tf.random.set_seed(seed)
    env = gym.make("FetchPickAndPlace-v1")
    env.seed(seed)
    np.random.seed(seed)
    test_set = []
    for i in range(TEST_EPS):
        state = env.reset()
        state, goal = utils.save_state(env)
        test_set.append((state, goal))

    states, actions = get_experience(INITIAL_TRAIN_EPS, env)
    print("Normal states, actions ", len(states), len(actions))

    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR)

    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - x.mean())/x.std()

    net.train(x, np.array(actions), BC_BS, BC_EPS)

    print("Normal learning results ", seed, " : ", test(net, test_set, env, xm, xs, False))
    
    ## Active Learning Part
    
    states, actions = states[:len(states)//2], actions[:len(actions)//2]
    
    for i in range((INITIAL_TRAIN_EPS//2)//ACTIVE_STEPS_RETRAIN):
    
        x = np.array(states)
        xm = x.mean()
        xs = x.std()
        x = (x - x.mean())/x.std()

        ae = AE(31, AE_HD, AE_HL, AE_LR)

        ae.train(x, AE_BS, AE_EPS)

        for j in range(ACTIVE_STEPS_RETRAIN):
            new_s, new_a = get_active_exp(env, ACTIVE_ERROR_THR, ae, xm, xs, False)
            states+=new_s
            actions+=new_a
        
    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - x.mean())/x.std()
    
    print("Active states, actions ", len(states), len(actions))
    
    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR)
    net.train(x, np.array(actions), BC_BS, BC_EPS)

    print("Active learning results ", seed, " : ",test(net, test_set, env, xm, xs, False))

        


if __name__ == "__main__":
    for k in range(5):
        go(k)

    