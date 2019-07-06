import tensorflow as tf
#from tensorflow.config.gpu import *
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime
from envs import *

import model
from ae import *
import man_controller
import utils
from hparams_perf import *


def get_experience(eps, env):
    global test_set
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        #Remove to not include in training set.
        state_, goal = utils.save_state(env)
        test_set.append((state_, goal))
        #####
        new_states, new_acts = man_controller.get_demo_cam2(env, state, norm = True)
        states+=new_states
        actions+=new_acts

    return states, actions

def test(model, test_set, env, xm, xs, am, ast, ae = None, render = False):
    successes, failures = 0,0
    errs_s, errs_f = 0, 0
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = utils.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        if ae:
            x = tf.keras.layers.Flatten()(model.call_hidden(np.concatenate((state[1], state[2][:,:,None]), -1)[None,:,:,:]))
            err = ae.error(x)
            print("Error ", err)
        else:
            err = model.error(np.concatenate((state[1], state[2][:,:,None]), -1)[None,:,:,:])
            print("Error ", err)
        picked = [False]
        for i in range(200):
            action = model(np.concatenate((state[1], state[2][:,:,None]), -1)[None,:,:,:])[0]

            new_state, *_ = env.step(action.numpy().reshape((4)))
         #   print(action)
            if render: env.render()
            state = new_state

            if not np.linalg.norm((state[0]["achieved_goal"]- state[0]["desired_goal"])) > 0.07:
        #        print("SUCCESS!")
                succeded = 1
                successes +=1
                errs_s += err
                break

        if not succeded:
            print("FAILURE")
            failures+=1
            errs_f += err

    print("Errors of successes: ",  errs_s/successes, ". Errors of failures: ", errs_f/failures)
    return successes, failures

def get_active_exp(env, threshold, ae, xm, xs, render):

    err_avg = 0
    for i in range(20):
        state = env.reset()
        error = ae.error(np.concatenate((state[1], state[2][:,:,None]), -1))
        err_avg+=error
    err_avg/=20

    state = env.reset()
    error = ae.error(np.concatenate((state[1], state[2][:,:,None]), -1))
    #print("predicted error", error)

    tried = 0
    while not error > threshold*err_avg:
        tried+=1
        state = env.reset()
        error = ae.error(np.concatenate((state[1], state[2][:,:,None]), -1))
  #      print("predicted error", error.numpy(), err_avg.numpy())
 #   print("Tried ", tried, " initial states")
    new_states, new_acts = man_controller.get_demo_cam2(env, state, render)

    return new_states, new_acts


def go(seed, file):
    global test_set
    global_start = time.time()
#    if not tf.__version__ == "2.0.0-beta0":
#        tf.random.set_random_seed(seed)
#    else:
    tf.random.set_seed(seed)
    env = CameraRobot(gym.make("FetchPickAndPlace-v1"), 75)
    env.seed(seed)
#    test_set = []  De-comment to reset test set
    for i in range(10):
        state = env.reset()
        state, goal = utils.save_state(env)
        test_set.append((state, goal))

    states, actions = get_experience(100, env)
    print("new states :", np.array(states).shape)
    print("Normal states, actions ", len(states), len(actions))

    net = model.ConvHybridNet2(75, 4, 4, 3, [5,3,3], [16,32,32], 64, 3, 0.001)
    #ae = net
    ae = AE(1152, 128, 3, 0.001, set_seed = seed)
    print("HERE")
    start = time.time()
    net.train(np.array(states), actions, 16, 400, network = "bc", print_loss = True)
    print("took ", str(time.time() - start))

    print("Creating cache")
    x_cache = net.create_cache(np.array(states), 64, 20, flatten = True)

    start = time.time()
    #net.train_ae(np.array(states), 64, 20, print_loss = True)
    ae.train(x_cache, 8, 10)
    print("took ", str(time.time() - start))
    result_t = test(net, test_set, env, 0, 1, 0, 1, render = False, ae = ae)
    print("Normal learning results ", seed, " : ", result_t)
    file.write(str("Normal learning results " + str(seed) + " : " + str(result_t)))
    print("Full experiment took ", (time.time() - global_start)/60, " minutes.")



if __name__ == "__main__":
    #tf.debugging.set_log_device_placement(True)
#    tf.config.gpu.set_per_process_memory_growth(True)
    test_set = []
    filename = "logs/" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
    with open(filename, "a+") as file:
        for k in range(1):
            print(str(hyperp))
            print(str(k))
            file.write(str(hyperp))
            file.write("\n\n")
            with tf.device("/device:CPU:0"):
                go(k, file)
