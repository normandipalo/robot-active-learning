import tensorflow as tf
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime
from fetch_env_two_obj.fetch.pick_and_place import FetchPickAndPlaceTwoEnv

import model
from ae import *
from future_model import *
import man_controller
import utils as u
from hparams import *
from envs import *

def robot_random_pick(env):
    #Pick the cube and bring arm to random pos.
    state = env.reset()
    state = man_controller.get_demo_cam_random_pick(env, state, norm = True, render = RENDER_ACT_EXP, stage = np.random.randint(0, 3))
    return state

def is_success(state):
    if not np.linalg.norm((state["achieved_goal"][:3]- state["desired_goal"])) > 0.07 and not np.linalg.norm((state["achieved_goal"][3:6]- state["achieved_goal"][:3])) > 0.07:
        return True
    return False

def robot_reset(env):
    random_act = np.random.randn(4)*0.3
    for i in range(20):
        state, *_ = env.step(random_act)
    return state

def avg_ae_error(ae, x):
    tot_error_trainset = 0
    for el in x:
        error = ae.error(el.reshape((1,-1)))
        tot_error_trainset+=error
    tot_error_trainset/=len(x)
    return tot_error_trainset

def get_experience(eps, env, render = False):
    states, actions = [], []
    for ep in range(eps):
        state = env.reset()
        #state = robot_reset(env) if ep%2==0 else robot_random_pick(env)
        #state = robot_reset(env)
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)
        states+=new_states
        actions+=new_acts

    return states, actions

def try_complete(model, ae, error_thr, env, xm, xs, am, ast, render = False):
    succeded = 0
    state, *_ = env.step([0.,0.,0.,0.])
    picked = [False]
#    print("Error threshold", error_thr)
    for i in range(100):
        error = ae.error((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)
#        print("Error in try_complete", error)
        time.sleep(0.2)

        if error > error_thr:
            #Return the current env and state to get an expert demo. Return False
            # to signal that it was unable to complete.
            return False, env, state, error

        action = model((np.concatenate((state["observation"],
                                    state["achieved_goal"],
                                    state["desired_goal"])).reshape((1,-1)) - xm)/xs)

        action = action*ast + am
        new_state, *_ = env.step(action[0])
     #   print(action)
        if render: env.render()
        state = new_state

        if is_success(state):
    #        print("SUCCESS!")
            return True, env, state, error

    return False, env, state, error


def test(model, test_set, env, xm, xs, am, ast, render = False):
    successes, failures = 0,0
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = u.set_state(env, test_set[i][0], test_set[i][1])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        for i in range(150):
            action = model((np.concatenate((state["observation"],
                                        state["achieved_goal"],
                                        state["desired_goal"])).reshape((1,-1)) - xm)/xs)

            action = action*ast + am
            new_state, *_ = env.step(action[0])
         #   print(action)
            if render: env.render()
            state = new_state

            if is_success(state):
        #        print("SUCCESS!")
                succeded = 1
                successes +=1

                break
        if not succeded:
       #     print("FAILURE")
            failures+=1
    return successes, failures

def get_active_exp2(env, avg_error_trainset, model, ae, xm, xs, am, ast, render, take_max = False, max_act_steps = 20):
    state = env.reset()
    state = robot_reset(env)
    succeded = True
    while succeded:
        state = env.reset()
        state = robot_reset(env)
        #Credo che dovrei resettare lo stato qua, altrimenti prova a completare sempre lo stesso.
        succeded, env, state, error = try_complete(model, ae, avg_error_trainset*ACTIVE_ERROR_THR, env, xm, xs, am, ast, render = RENDER_TEST)
    #Here we have the env and the state where the robot doesn't know what to do.
    time.sleep(1.)
#    print("Expert demo.")
    new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)

#    if len(new_states) > 100:
        #If it's so long the demo failed.
        #Should consider to reset it and try again, otherwise we waste a demo.
#        return [], []

    #Recursively call until a demo works.
    while len(new_states) > 150:
        print("retry")
        #If it's so long the demo failed.
        #Should consider to reset it and try again, otherwise we waste a demo.
        new_states, new_acts = get_active_exp2(env, avg_error_trainset, model, ae, xm, xs, am, ast, render, take_max = False, max_act_steps = 20)
    return new_states, new_acts

def get_active_exp(env, threshold, ae, xm, xs, render, take_max = False, max_act_steps = 20):
    err_avg = 0
    for i in range(20):
        state = env.reset()
        state = robot_reset(env) if i%2==0 else robot_random_pick(env)
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

    k = True
    if not take_max:
        tried = 0
        while not error > threshold*err_avg:
            k = not k
            tried+=1
            state = env.reset()
            #state = robot_reset(env)
            state = robot_reset(env) if k else robot_random_pick(env)
            error = ae.error((np.concatenate((state["observation"],
                                            state["achieved_goal"],
                                            state["desired_goal"])).reshape((1,-1)) - xm)/xs)
      #      print("predicted error", error.numpy(), err_avg.numpy())
        print("Tried ", tried, " initial states")
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)

        return new_states, new_acts

    else:
        errs_states = []
        for k in range(max_act_steps):
            state = env.reset()
            state = robot_reset(env) if k%2==0 else robot_random_pick(env)
            error = ae.error((np.concatenate((state["observation"],
                                            state["achieved_goal"],
                                            state["desired_goal"])).reshape((1,-1)) - xm)/xs)
            s, g = u.save_state(env)
            errs_states.append([s, g, error])

        max_error = -1000
        max_key = ()
        for el in range(len(errs_states)):
            if errs_states[el][2] > max_error:
                max_error = errs_states[el][2]
                max_key = el

        new_env = u.set_state(env, errs_states[max_key][0], errs_states[max_key][1])
        state, *_ = new_env.step(np.zeros(4))

        new_states, new_acts = man_controller.get_demo(new_env, state, CTRL_NORM, render)

        while len(new_states) > 150:
            print(len(new_states))
            print("retry")
            #If it's so long the demo failed.
            #Should consider to reset it and try again, otherwise we waste a demo.
            new_states, new_acts = get_active_exp(env, threshold, ae, xm, xs, render, take_max, max_act_steps)

        return new_states, new_acts


def go(seed, file):

    tf.random.set_seed(seed)
    env = FetchPickAndPlaceTwoEnv()
    env = Fetch2Cubes(env)
    env.seed(seed)
    np.random.seed(seed)
    test_set = []
    for i in range(TEST_EPS):
        state = env.reset()
        state = robot_reset(env)
        state, goal = u.save_state(env)
        test_set.append((state, goal))

    states, actions = get_experience(INITIAL_TRAIN_EPS, env, RENDER_TEST)
    print("Normal states, actions ", len(states), len(actions))

    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - xm)/xs

    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - am)/ast

    start = time.time()
    print(x.shape)
    net.train(x, a, BC_BS, BC_EPS)
    print("Training took:")
    print(time.time() - start)
    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("Normal learning results ", seed, " : ", result_t)
    file.write(str("Normal learning results " + str(seed) + " : " + str(result_t)))

    ## Active Learning Part ###
    tf.random.set_seed(seed)
    env = FetchPickAndPlaceTwoEnv()
    env = Fetch2Cubes(env)
    env.seed(seed)
    np.random.seed(seed)

    """

    states, actions = states[:math.floor(len(states)*ORG_TRAIN_SPLIT)], actions[:math.floor(len(actions)*ORG_TRAIN_SPLIT)]
    #Train behavior net on half data.
    net_hf = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - xm)/xs

    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - am)/ast
    net_hf.train(x, a, BC_BS, BC_EPS*2)

    #get_experience(int(INITIAL_TRAIN_EPS*ORG_TRAIN_SPLIT), env)
    act_l_loops = math.ceil(((1.-ORG_TRAIN_SPLIT)*INITIAL_TRAIN_EPS)//ACTIVE_STEPS_RETRAIN)
    if act_l_loops == 0: act_l_loops+=1
    for i in range(act_l_loops):

        x = np.array(states)
        xm = x.mean()
        xs = x.std()
        x = (x - xm)/xs

        a = np.array(actions)
        am = a.mean()
        ast = a.std()
        a = (a - am)/ast

        #if AE_RESTART: ae = DAE(31, AE_HD, AE_HL, AE_LR, set_seed = seed)
        #Reinitialize both everytime and retrain.

        obs_dim, act_dim = len(x[0]), len(a[0])
        norm = Normalizer(obs_dim, act_dim).fit(x[:-1], a[:-1], x[1:])

        start = time.time()
        dyn = NNDynamicsModel(obs_dim, act_dim, 128, norm, 64, 100, 3e-4)
        dyn.fit({"states": x[:-1], "acts" : a[:-1], "next_states" : x[1:]}, plot = False)
        print("Dynamics model trained in", time.time() - start)

        ae_x = AE(states[0].shape[0], AE_HD, AE_HL, AE_LR, set_seed = seed)
        #ae = RandomNetwork(1, AE_HD, AE_HL, AE_LR)
        ae_x.train(x, AE_BS, AE_EPS)
        ae = FutureUnc(net, dyn, ae_x, steps = 5)
        net_hf = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
        start = time.time()
    #    ae.train(x, AE_BS, AE_EPS)
        net_hf.train(x, a, BC_BS, BC_EPS*2)
        avg_error = avg_ae_error(ae, x)

        print("Training took:")
        print(time.time() - start)

        for j in range(ACTIVE_STEPS_RETRAIN):
            #new_s, new_a = get_active_exp(env, ACTIVE_ERROR_THR, ae, xm, xs, RENDER_ACT_EXP, TAKE_MAX, MAX_ACT_STEPS)
            new_s, new_a = get_active_exp2(env, avg_error, net_hf, ae, xm, xs, am, ast, RENDER_ACT_EXP, TAKE_MAX, MAX_ACT_STEPS)
    #        print("len new s ", len(new_s), " len new a ", len(new_a))
            states+=new_s
            actions+=new_a

    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - xm)/xs

    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - am)/ast

    print("Active states, actions ", len(states), len(actions))

    net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
    net.train(x, a, BC_BS, BC_EPS)

    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("Active learning results ", seed, " : ", result_t)
    file.write(str("Active learning results " + str(seed) + " : " + str(result_t)))

    """
    #print("Active learning results ", seed, " : ",test(net, test_set, env, xm, xs, am, ast, True))

    start = time.time()
    norm = Normalizer(49, 4).fit(x[:-1], a[:-1], x[1:])

    dyn = NNDynamicsModel(49, 4, 128, norm, 64, 2000, 3e-4)
    dyn.fit({"states": x[:-1], "acts" : a[:-1], "next_states" : x[1:]}, plot = False)

    ae_x = DAE(49, AE_HD, AE_HL, AE_LR)
    ae_x.train(x, AE_BS, AE_EPS)
    net = model.BCPlanner(net, dyn, ae_x, 4, W1, W2, W)
    print("Training took:")
    print(time.time() - start)

    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("PLANNING: Active learning results ", seed, " : ", result_t)
    file.write(str("PLANNING: Active learning results " + str(seed) + " : " + str(result_t)))



if __name__ == "__main__":

    filename = "logs/" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ".txt"
    print(filename)
    with open(filename, "a+") as file:
        print(str(hyperp))
        file.write(str(hyperp))
        for k in range(0,12):
            print(str(k))
            file.write("\n" + str(k))
            file.write("\n\n")
            with tf.device("/device:CPU:0"):
                go(k, file)
