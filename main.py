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
from hparams import *

def robot_reset(env):
    random_act = np.random.randn(4)*0.3
    for i in range(20):
        state, *_ = env.step(random_act)
    return state

def get_experience(eps, env):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        state = robot_reset(env)
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM)
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
        for i in range(100):
            action = model((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)

            action = action*ast + am
            new_state, *_ = env.step(action[0])
         #   print(action)
            if render: env.render()
            state = new_state

            if not np.linalg.norm((state["achieved_goal"]- state["desired_goal"])) > 0.10:
        #        print("SUCCESS!")
                succeded = 1
                successes +=1

                break
        if not succeded:
       #     print("FAILURE")
            failures+=1
    return successes, failures

def get_active_exp(env, threshold, ae, xm, xs, render, take_max = False, max_act_steps = 20):

    err_avg = 0
    for i in range(20):
        state = env.reset()
        state = robot_reset(env)
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

    if not take_max:
        tried = 0
        while not error > threshold*err_avg:
            tried+=1
            state = env.reset()
            state = robot_reset(env)
            error = ae.error((np.concatenate((state["observation"],
                                            state["achieved_goal"],
                                            state["desired_goal"])).reshape((1,-1)) - xm)/xs)
      #      print("predicted error", error.numpy(), err_avg.numpy())
     #   print("Tried ", tried, " initial states")
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)

        return new_states, new_acts

    else:
        errs_states = []
        for k in range(max_act_steps):
            state = env.reset()
            state = robot_reset(env)
            error = ae.error((np.concatenate((state["observation"],
                                            state["achieved_goal"],
                                            state["desired_goal"])).reshape((1,-1)) - xm)/xs)
            s, g = utils.save_state(env)
            errs_states.append([s, g, error])

        max_error = -1000
        max_key = ()
        for el in range(len(errs_states)):
            if errs_states[el][2] > max_error:
                max_error = errs_states[el][2]
                max_key = el

        new_env = utils.set_state(env, errs_states[max_key][0], errs_states[max_key][1])
        state, *_ = new_env.step(np.zeros(4))

        new_states, new_acts = man_controller.get_demo(new_env, state, CTRL_NORM, render)

        return new_states, new_acts


def go(seed, file):
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
        state = robot_reset(env)
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

    start = time.time()
    net.train(x, a, BC_BS, BC_EPS)
    print("Training took:")
    print(time.time() - start)
    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("Normal learning results ", seed, " : ", result_t)
    file.write(str("Normal learning results " + str(seed) + " : " + str(result_t)))

    ## Active Learning Part ###
    if not tf.__version__ == "2.0.0-alpha0":
        tf.random.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)
    env = gym.make("FetchPickAndPlace-v1")
    env.seed(seed)
    np.random.seed(seed)

    states, actions = states[:math.floor(len(states)*ORG_TRAIN_SPLIT)], actions[:math.floor(len(actions)*ORG_TRAIN_SPLIT)]
    #get_experience(int(INITIAL_TRAIN_EPS*ORG_TRAIN_SPLIT), env)
    act_l_loops = math.ceil(((1.-ORG_TRAIN_SPLIT)*INITIAL_TRAIN_EPS)//ACTIVE_STEPS_RETRAIN)
    if act_l_loops == 0: act_l_loops+=1
    ae = DAE(31, AE_HD, AE_HL, AE_LR, set_seed = seed)
    for i in range(act_l_loops):

        x = np.array(states)
        xm = x.mean()
        xs = x.std()
        x = (x - x.mean())/x.std()

        if AE_RESTART: ae = DAE(31, AE_HD, AE_HL, AE_LR, set_seed = seed)

        start = time.time()
        ae.train(x, AE_BS, AE_EPS)
        print("Training took:")
        print(time.time() - start)

        for j in range(ACTIVE_STEPS_RETRAIN):
            new_s, new_a = get_active_exp(env, ACTIVE_ERROR_THR, ae, xm, xs, RENDER_ACT_EXP, TAKE_MAX, MAX_ACT_STEPS)
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

    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("Active learning results ", seed, " : ", result_t)
    file.write(str("Active learning results " + str(seed) + " : " + str(result_t)))

    #print("Active learning results ", seed, " : ",test(net, test_set, env, xm, xs, am, ast, True))




if __name__ == "__main__":

    filename = "logs/" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
    with open(filename, "a+") as file:
        print(str(hyperp))
        file.write(str(hyperp))
        for k in range(0,50):
            print(str(k))
            file.write("\n" + str(k))
            file.write("\n\n")
            with tf.device("/device:CPU:0"):
                go(k, file)
