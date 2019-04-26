import tensorflow as tf
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime

import model
from ae import AE
import man_controller
import utils

"""INITIAL_TRAIN_EPS = 200

BC_LR = 1e-3
BC_HD = 128
BC_HL = 2
BC_BS = 64
BC_EPS = 250

AE_HD = 128
AE_HL = 2
AE_LR = 1e-3
AE_BS = 64
AE_EPS = 50

TEST_EPS = 200
ACTIVE_STEPS_RETRAIN = 10
ACTIVE_ERROR_THR = 0.

ORG_TRAIN_SPLIT = 0.75
"""

from hparams import *

def get_experience(eps, env):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        new_states, new_acts = man_controller.get_demo(env, state)
        states+=new_states
        actions+=new_acts
                
    return states, actions

def test(model, test_set, env, xm, xs, am, ast, render = False):
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

            action = action*ast + am
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


def go(seed, file):
    if not tf.__version__ == "2.0.0-alpha0":
        tf.random.set_random_seed(seed)
    else: 
        tf.random.set_seed(seed)
    env = gym.make("FetchPickAndPlace-v1")
    env.seed(seed)
    test_set = []
    for i in range(TEST_EPS):
        state = env.reset()
        state, goal = utils.save_state(env)
        test_set.append((state, goal))

    states, actions = get_experience(INITIAL_TRAIN_EPS, env)
    print("Normal states, actions ", len(states), len(actions))

    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)

    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - x.mean())/x.std()
    
    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - a.mean())/a.std()

    net.train(x, a, BC_BS, BC_EPS)
    
    result_t = test(net, test_set, env, xm, xs, am, ast, False)
    print("Normal learning results ", seed, " : ", result_t)
    file.write(str("Normal learning results " + str(seed) + " : " + str(result_t)))
    
    ## Active Learning Part ###
    if not tf.__version__ == "2.0.0-alpha0":
        tf.random.set_random_seed(seed)
    else: 
        tf.random.set_seed(seed)
    env = gym.make("FetchPickAndPlace-v1")
    env.seed(seed)
    
    states, actions = states[:math.floor(len(states)*ORG_TRAIN_SPLIT)], actions[:math.floor(len(actions)*ORG_TRAIN_SPLIT)]
    
  
    for i in range(int(((1-ORG_TRAIN_SPLIT)*INITIAL_TRAIN_EPS)//ACTIVE_STEPS_RETRAIN)):
    
        x = np.array(states)
        xm = x.mean()
        xs = x.std()
        x = (x - x.mean())/x.std()

        ae = AE(31, AE_HD, AE_HL, AE_LR, set_seed = seed)

        ae.train(x, AE_BS, AE_EPS)

        for j in range(ACTIVE_STEPS_RETRAIN):
            new_s, new_a = get_active_exp(env, ACTIVE_ERROR_THR, ae, xm, xs, False)
            states+=new_s
            actions+=new_a
        
    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - x.mean())/x.std()
    
    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - a.mean())/a.std()
    
    print("Active states, actions ", len(states), len(actions))
    
    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
    net.train(x, a, BC_BS, BC_EPS)

    result_t = test(net, test_set, env, xm, xs, am, ast, False)
    print("Active learning results ", seed, " : ", result_t)
    file.write(str("Active learning results " + str(seed) + " : " + str(result_t)))
    
    #print("Active learning results ", seed, " : ",test(net, test_set, env, xm, xs, am, ast, True))

        


if __name__ == "__main__":

    filename = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
    with open(filename, "a+") as file:
        for k in range(5):
            print(str(hyperp))
            print(str(k))
            file.write(str(hyperp))
            file.write("\n\n")
            go(k, file)

    
