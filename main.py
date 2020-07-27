import tensorflow as tf
import gym
import numpy as np
import math
import time
import datetime
from time import gmtime, strftime
from metaworld.envs.mujoco.sawyer_xyz import *
import sawyer_test

import model
from ae import *
import man_controller
import utils
from hparams import *
from future_model import *


def robot_reset(env):
    random_act = np.random.randn(4)*0.5
    random_act[3] = 0
    for i in range(20):
        state, *_ = env.step(random_act)
        if RENDER_ACT_EXP: env.render()
    return state

def avg_ae_error(ae, x):
    tot_error_trainset = 0
    for el in x:
        error = ae.error(el.reshape((1,-1)))
        tot_error_trainset+=error
    tot_error_trainset/=len(x)
    return tot_error_trainset

def get_experience(eps, env, render):
    states, actions = [], []
    ep = 0
    while ep < eps:
        state = env.reset()
        state = robot_reset(env)
        p_peg = utils.save_state(env)
        dir = np.zeros(4)
        dir[:3] = np.random.randn(3)*0.5
        dir[2] = np.linalg.norm(dir[2])
    #    for k in range(20):
    #        env.step(dir)
    #        env.render()
        new_states, new_acts, success = man_controller.get_demo(env, state, CTRL_NORM, render)
        if success:
        #    test_set.append((p_peg))
            ep+=1
            states+=new_states
            actions+=new_acts
    #    if not success:
    #        print("FAIL")
    return states, actions

def try_complete(model, ae, error_thr, env, xm, xs, am, ast, render = False):
    succeded = 0
    state, *_ = env.step([0.,0.,0.,0.])
    picked = [False]
#    print("Error threshold", error_thr)
    for i in range(400):
        error = ae.error((state[None] - xm)/xs)
    #    time.sleep(0.2)

        if error > error_thr:
            #Return the current env and state to get an expert demo. Return False
            # to signal that it was unable to complete.
            print("Uncertainty surpassed threshold.")
            #input("go on")
            return False, env, state, error
        action = model((state[None] - xm)/xs)
        action = action*ast + am
        new_state, *_ = env.step(np.nan_to_num(action[0]))
     #   print(action)
        if render: env.render()
        state = new_state

        if sawyer_test.check_success(env, state):
    #        print("SUCCESS!")
            return True, env, state, error

    return False, env, state, error


def test(model, test_set, env, xm, xs, am, ast, render = False):
    successes, failures = 0,0
    for i in range(len(test_set)):
        succeded = 0
        env.reset()
        env = utils.set_state(env, test_set[i])
        state, *_ = env.step([0.,0.,0.,0.])
        picked = [False]
        for i in range(500):
            action = model((state[None] - xm)/xs)

            action = action*ast + am
            new_state, *_ = env.step(np.nan_to_num(action[0]))
            if render: env.render()
            state = new_state

            if sawyer_test.check_success(env, state):
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
        succeded, env, state, error = try_complete(model, ae, avg_error_trainset*ACTIVE_ERROR_THR, env, xm, xs, am, ast, render = RENDER_ACT_EXP)
    #Here we have the env and the state where the robot doesn't know what to do.
#    print("Expert demo.")

    #Go slightly up.
    #for i in range(10):
#        state, *_ = env.step(np.array([0.,0.,0.5,-1.])) #if state[5] < 0.03 else env.step(np.array([0.,0.,0.5,1.]))
    new_states, new_acts, success = man_controller.get_demo(env, state, CTRL_NORM, render = RENDER_ACT_EXP)

#    if len(new_states) > 100:
        #If it's so long the demo failed.
        #Should consider to reset it and try again, otherwise we waste a demo.
#        return [], []

    #Recursively call until a demo works.
    while len(new_states) >= 999:
        print("other round", len(new_states))
        #If it's so long the demo failed.
        #Should consider to reset it and try again, otherwise we waste a demo.
        new_states, new_acts = get_active_exp2(env, avg_error_trainset, model, ae, xm, xs, am, ast, render = RENDER_ACT_EXP, take_max = False, max_act_steps = 20)
    return new_states, new_acts

def get_active_exp(env, threshold, ae, xm, xs, render, take_max = False, max_act_steps = 20):

    err_avg = 0
    for i in range(20):
        state = env.reset()
        state = robot_reset(env)
        error = ae.error((state[None] - xm)/xs)
        err_avg+=error
    err_avg/=20

    state = env.reset()
    error = ae.error((state[None] - xm)/xs)
    #print("predicted error", error)

    if not take_max:
        tried = 0
        while not error > threshold*err_avg:
            tried+=1
            state = env.reset()
            state = robot_reset(env)
            error = ae.error((state[None] - xm)/xs)
      #      print("predicted error", error.numpy(), err_avg.numpy())
     #   print("Tried ", tried, " initial states")
        new_states, new_acts = man_controller.get_demo(env, state, CTRL_NORM, render)

        return new_states, new_acts

    else:
        errs_states = []
        for k in range(max_act_steps):
            state = env.reset()
            state = robot_reset(env)
            error = ae.error((state[None] - xm)/xs)
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

    tf.random.set_seed(seed)
    env = SawyerNutAssemblyEnv() #goal_low = (-0.15,0.5,0.1), goal_high = (0.15,0.8,0.1))
    env.seed(seed)
    np.random.seed(seed)
    test_set  = []
    for i in range(TEST_EPS):
        state = env.reset()
        p_peg = utils.save_state(env)
        test_set.append((p_peg))

    states, actions = get_experience(INITIAL_TRAIN_EPS, env, RENDER_ACT_EXP)
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

    if DISABLE_NORM:
        #Looks like these are creating instabilities.
        am, xm = 0., 0.
        ast, xs = 1., 1.

    start = time.time()
    print(x.shape)
    net.train(x, a, BC_BS, BC_EPS)
    print("Training took:")
    print(time.time() - start)
    #input("NL TEST")
    result_t = test(net, test_set, env, xm, xs, am, ast, RENDER_TEST)
    print("Normal learning results ", seed, " : ", result_t)
    file.write(str("Normal learning results " + str(seed) + " : " + str(result_t)))
    
    ## Active Learning Part ###
    tf.random.set_seed(seed)
    env = SawyerNutAssemblyEnv() #goal_low = (-0.15,0.5,0.1), goal_high = (0.15,0.8,0.1))
    env.seed(seed)
    np.random.seed(seed)

    states, actions = states[:math.floor(len(states)*ORG_TRAIN_SPLIT)], actions[:math.floor(len(actions)*ORG_TRAIN_SPLIT)]
    #Train behavior net on half data.
    net_hf = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
    x = np.array(states)
    xm = x.mean()
    xs = x.std()
    x = (x - x.mean())/x.std()

    a = np.array(actions)
    am = a.mean()
    ast = a.std()
    a = (a - a.mean())/a.std()
    if DISABLE_NORM:
        #Looks like these are creating instabilities.
        am, xm = 0., 0.
        ast, xs = 1., 1.

    net_hf.train(x, a, BC_BS, BC_EPS)

    #get_experience(int(INITIAL_TRAIN_EPS*ORG_TRAIN_SPLIT), env)
    act_l_loops = math.ceil(((1.-ORG_TRAIN_SPLIT)*INITIAL_TRAIN_EPS)//ACTIVE_STEPS_RETRAIN)
    if act_l_loops == 0: act_l_loops+=1
    for i in range(act_l_loops):

        x = np.array(states)
        xm = x.mean()
        xs = x.std()
        x = (x - x.mean())/x.std()

        a = np.array(actions)
        am = a.mean()
        ast = a.std()
        a = (a - a.mean())/a.std()

        if DISABLE_NORM:
            #Looks like these are creating instabilities.
            am, xm = 0., 0.
            ast, xs = 1., 1.

        #if AE_RESTART: ae = DAE(31, AE_HD, AE_HL, AE_LR, set_seed = seed)
        #Reinitialize both everytime and retrain.
        norm = Normalizer(9, 4).fit(x[:-1], a[:-1], x[1:])

        dyn = NNDynamicsModel(9, 4, 128, norm, 64, 5, 3e-4)
        dyn.fit({"states": x[:-1], "acts" : a[:-1], "next_states" : x[1:]}, plot = False)

        ae_x = AE(9, AE_HD, AE_HL, AE_LR)
        #ae = RandomNetwork(1, AE_HD, AE_HL, AE_LR)

        ae_x.train(x, AE_BS, AE_EPS)
     #   ae = ae_x
        ae = FutureUnc(net, dyn, ae_x, steps = 5)
        net_hf = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)

        start = time.time()

        net_hf.train(x, a, BC_BS, BC_EPS, show_loss = False)
        avg_error = avg_ae_error(ae, x)

        print("Training took:")
        print(time.time() - start)

        #input("Getting Active Experience")
        for j in range(ACTIVE_STEPS_RETRAIN):
            #new_s, new_a = get_active_exp(env, ACTIVE_ERROR_THR, ae, xm, xs, RENDER_ACT_EXP, TAKE_MAX, MAX_ACT_STEPS)
            new_s, new_a = get_active_exp2(env, avg_error, net_hf, ae, xm, xs, am, ast, RENDER_TEST, TAKE_MAX, MAX_ACT_STEPS)
            #print("len new s ", len(new_s), " len new a ", len(new_a))
    #        print("Added new stuff.", len(new_s))
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

    if DISABLE_NORM:
        #Looks like these are creating instabilities.
        am, xm = 0., 0.
        ast, xs = 1., 1.

    print("Active states, actions ", len(states), len(actions))

    for ep_m in [1.]:
        print("epochs", math.ceil(BC_EPS*ep_m))
        net = model.BCModel(states[0].shape[0], actions[0].shape[0], BC_HD, BC_HL, BC_LR, set_seed = seed)
        net.train(x, a, BC_BS, math.ceil(BC_EPS*ep_m), show_loss = False)

        #input("AL TEST")
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
            file.flush()
