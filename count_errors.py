import tensorflow as tf
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime

import model
from ae import AE, DAE
import man_controller
import utils

hyperp = {"INITIAL_TRAIN_EPS" : 50,

"BC_LR" : 1e-3,
"BC_HD" : 128,
"BC_HL" : 2,
"BC_BS" : 64,
"BC_EPS" : 200,

"AE_HD" : 32,
"AE_HL" : 2,
"AE_LR" : 1e-3,
"AE_BS" : 32,
"AE_EPS" : 10,

"TEST_EPS" : 500,
"ACTIVE_STEPS_RETRAIN" : 10,
"ACTIVE_ERROR_THR" : 1.5,

"ORG_TRAIN_SPLIT" : 1.}

INITIAL_TRAIN_EPS = hyperp["INITIAL_TRAIN_EPS"]
BC_LR = hyperp["BC_LR"]
BC_HD = hyperp["BC_HD"]
BC_HL = hyperp["BC_HL"]
BC_BS = hyperp["BC_BS"]
BC_EPS = hyperp["BC_EPS"]

AE_HD = hyperp["AE_HD"]
AE_HL = hyperp["AE_HL"]
AE_LR = hyperp["AE_LR"]
AE_BS = hyperp["AE_BS"]
AE_EPS = hyperp["AE_EPS"]

TEST_EPS = hyperp["TEST_EPS"]
ACTIVE_STEPS_RETRAIN = hyperp["ACTIVE_STEPS_RETRAIN"]
ACTIVE_ERROR_THR = hyperp["ACTIVE_ERROR_THR"]

ORG_TRAIN_SPLIT = hyperp["ORG_TRAIN_SPLIT"]

#from hparams import *

def get_experience(eps, env):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        new_states, new_acts = man_controller.get_demo(env, state)
        states+=new_states
        actions+=new_acts
                
    return states, actions

def test(model, ae, test_set, env, xm, xs, am, ast, render = False):
    successes, failures = 0,0
    error_succ, error_fail = 0.,0.
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = utils.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
        #print("Uncertainty ", error.numpy())
        for i in range(100):
            action = model((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)

            action = action*ast + am
            new_state, *_ = env.step(action[0])
         #   print(action)
            if render: env.render()
            state = new_state
       
            if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.07:
          #      print("SUCCESS!")
                succeded = 1
                successes +=1
                error_succ += error

                break
        if not succeded: 
         #   print("FAILURE")
            failures+=1
            error_fail += error
          
    return successes, failures, error_succ/successes, error_fail/failures

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
    
    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - a.mean())/a.std()

    net.train(x, a, BC_BS, BC_EPS)

    ae = DAE(31, AE_HD, AE_HL, AE_LR)

    ae.train(x, AE_BS, AE_EPS)

    succ, fail, error_avg_s, error_avg_f = test(net, ae, test_set, env, xm, xs, am, ast, False)
    print("Active learning results ", seed, " : ", succ, fail, error_avg_s, error_avg_f)
   # file.write(str("Active learning results " + str(seed) + " : " + str(result_t)))
    
    #print("Active learning results ", seed, " : ",test(net, test_set, env, xm, xs, am, ast, True))

        


if __name__ == "__main__":

    filename = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
   # with open(filename, "a+") as file:
    for k in range(10):
        print(str(hyperp))
        print(str(k))
       
        go(k)

    