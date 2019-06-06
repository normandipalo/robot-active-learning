import gym
import numpy as np

def save_state(env):
    try:
        goal = env.env.goal
        state = env.env.sim.get_state()
    except:
        goal = env.env.env.goal
        state = env.env.env.sim.get_state()
    return state, goal

def set_state(env, state, goal):
    try:
        env.env.sim.set_state(state)
        env.env.goal = goal
    except:
        env.env.env.sim.set_state(state)
        env.env.env.goal = goal
    return env

get_state = save_state
