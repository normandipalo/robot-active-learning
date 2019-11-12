import gym
import numpy as np
import tensorflow as tf

class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.input_dim = input_dim
        self.mean_v = tf.Variable(initial_value = tf.zeros(1, dtype = tf.float64), trainable = True, dtype = tf.float64)
        self.std_v = tf.Variable(initial_value = tf.ones(1, dtype = tf.float64), trainable = True, dtype = tf.float64)

    @tf.function
    def call(self, x):
        mean = tf.math.reduce_mean(x) #.numpy().mean()
        std = tf.math.reduce_std(x) #.numpy().std()
        return ((x - mean)/std)*self.std_v + self.mean_v


def save_state(env):
    try:
        goal = env.goal
        state = env.sim.get_state()
    except:
        goal = env.env.env.goal
        state = env.env.env.sim.get_state()
    return state, goal

def set_state(env, state, goal):
    try:
        env.sim.set_state(state)
        env.goal = goal
    except:
        env.env.env.sim.set_state(state)
        env.env.env.goal = goal
    return env

get_state = save_state
