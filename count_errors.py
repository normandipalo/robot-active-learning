import tensorflow as tf
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime

import model
from ae import *
import man_controller
import utils

hyperp = {"INITIAL_TRAIN_EPS" : 30,

"BC_LR" : 1e-3,
"BC_HD" : 128,
"BC_HL" : 2,
"BC_BS" : 64,
"BC_EPS" : 400,

"AE_HD" : 256,
"AE_HL" : 1,
"AE_LR" : 1e-2,
"AE_BS" : 64,
"AE_EPS" : 20,

"TEST_EPS" : 100,
"ACTIVE_STEPS_RETRAIN" : 10,
"ACTIVE_ERROR_THR" : 1.5,

"ORG_TRAIN_SPLIT" : 1.,
"FULL_TRAJ_ERROR" : False,
"CTRL_NORM" : True,
"RENDER_TEST" : False}

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
FULL_TRAJ = hyperp["FULL_TRAJ_ERROR"]
CTRL_NORM = hyperp["CTRL_NORM"]

TEST_EPS = hyperp["TEST_EPS"]
ACTIVE_STEPS_RETRAIN = hyperp["ACTIVE_STEPS_RETRAIN"]
ACTIVE_ERROR_THR = hyperp["ACTIVE_ERROR_THR"]

ORG_TRAIN_SPLIT = hyperp["ORG_TRAIN_SPLIT"]
RENDER_TEST = hyperp["RENDER_TEST"]

#from hparams import *

def get_experience(eps, env):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM)
        states+=new_states
        actions+=new_acts
                
    return states, actions

def test(model, ae, test_set, env, xm, xs, am, ast, fulltraj = False, render = False):
    successes, failures = 0,0
    error_succ, error_fail = 0.,0.
    succ_errors_list, fail_errors_list = [], []
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = utils.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        
        error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
        tot_error = error
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
            if fulltraj:
            #If I choose to sum up the errors of the entire trajectory I integrate errors
            # on tot_error
                error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
                tot_error+=error

            if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.07:
          #      print("SUCCESS!")
                succeded = 1
                successes +=1
                
                #divide by number of steps to get an average
                if fulltraj:
                    error_succ += tot_error/(i + 1)
                    succ_errors_list.append(tot_error/(i + 1))
                else:
                    error_succ += tot_error
                    succ_errors_list.append(tot_error)
                break
        if not succeded: 
         #   print("FAILURE")
            failures+=1
            #divide by number of steps to get an average
            if fulltraj:
                error_fail += tot_error/(i + 1)
                fail_errors_list.append(tot_error/(i+1))
            else:
                error_fail += tot_error
                fail_errors_list.append(tot_error)
                
    return successes, failures, error_succ/successes, error_fail/failures, succ_errors_list, fail_errors_list
    
def predict(model, ae, test_set, env, xm, xs, am, ast, tot_error_trainset):
    succ_tp, succ_fp, succ_tn, succ_fn = 0,0,0,0
    successes, failures = 0, 0
    for i in range(len(test_set)):
        
        env.reset()
        env = utils.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        succeded = False
        error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
                            
        if error > 2.3*tot_error_trainset:
            prediction = "failure"
        else:
            prediction = "success"
      #  prediction = "success" #assume you think you'll always succeed
            
        for i in range(100):
            action = model((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)

            action = action*ast + am
            new_state, *_ = env.step(action[0])
         #   print(action)
            
            state = new_state
       

            if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.07:
          #      print("SUCCESS!")
                successes += 1
                succeded = True
                if prediction == "success":
                    succ_tp +=1
                elif prediction == "failure":
                    succ_fn +=1
                break
                
                #divide by number of steps to get an average

        if not succeded: 
            failures += 1
            if prediction == "failure":
                succ_tn +=1
            elif prediction == "success":
                succ_fp += 1
                
    print("successes, failures", successes, failures)
    return succ_tp, succ_fp, succ_tn, succ_fn
        
    

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
    new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)

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
    file.write("Normal states, actions " + str(len(states)) + str(len(actions)))
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

    #ae = AE(31, AE_HD, AE_HL, AE_LR)
    ae = RandomNetwork(1, AE_HD, AE_HL, AE_LR)

    ae.train(x, AE_BS, AE_EPS)
    
    tot_error_trainset = 0
    for el in x:
        error = ae.error(el.reshape((1,-1)))
        tot_error_trainset+=error
    tot_error_trainset/=len(x)
    
    print("Average error on train set", tot_error_trainset)
    file.write(str("Average error on train set") +str(tot_error_trainset))
    
    # I try to estimate the error on full train trajectories by multiplying
    # the average by len(dataset)/num_episodes, that is basically
    # the average episode lenght.
    
    if FULL_TRAJ:
        tot_error_train_fulltraj = 0
        avg_ep_lenght = len(x)/INITIAL_TRAIN_EPS
        tot_error_train_fulltraj = tot_error_trainset*avg_ep_lenght
    
        print("Average full trajectory error on train set", tot_error_train_fulltraj)
        file.write(str("Average full trajectory error on train set") + str(tot_error_train_fulltraj))
    succ, fail, error_avg_s, error_avg_f, succ_list, fail_list = test(net, ae, test_set, env, xm, xs, am, ast, fulltraj = FULL_TRAJ, render = RENDER_TEST)
    print("Active learning results ", seed, " : ", succ, fail, "avg error on succ trails: ", error_avg_s, "on fail: ", error_avg_f, "std on succ:", np.std(succ_list), "on fail:", np.std(fail_list))
    file.write(str("Active learning results ") + str(seed) +  str(" : ") + str(succ) + str(fail) + str(error_avg_s) + str(error_avg_f) + str(np.std(succ_list)) +  str(np.std(fail_list)))
  
  #  file.write(str("Active learning results " + str(seed) + " : " + str(result_t)))
    
    
    
    succ_tp, succ_fp, succ_tn, succ_fn = predict(net, ae, test_set, env, xm, xs, am, ast, tot_error_trainset)
    
    #change to consider failures
    #succ_tp, succ_fp, succ_tn, succ_fn = succ_tn, succ_fn, succ_tp, succ_fp
    
    print("succ tp, fp, tn, fn", succ_tp, succ_fp, succ_tn, succ_fn)
    precision = (succ_tp/(succ_tp+succ_fp + 0.001))
    recall = (succ_tp/(succ_tp+succ_fn))
    print("F1 score", (precision*recall/(precision + recall)))
        


if __name__ == "__main__":

    filename = "logs/counterrors" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
    with open(filename, "a+") as file:
        for k in range(30):
            print(str(hyperp))
            print(str(k))
            file.write(str(hyperp))
            file.write("\n\n")
            go(k)

    
