import numpy as np
import copy
import sawyer_test

def get_demo(env, state, norm = False, render = False):
    states, actions = [], []
    picked = [False]
    check_bits = np.zeros(4)
    success = False
    for i in range(400):
        cb = sawyer_test.check(env, state, check_bits)
        action, steps = sawyer_test.get_act(env, state, cb, norm)
        for s in range(steps):
            new_state, *_ = env.step(action)
            if render: env.render()
            states.append(state)
            actions.append(action)
            state = new_state
        if sawyer_test.check_success(env, state):
            success = True
            break
    return states, actions, success
