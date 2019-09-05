import sys, os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#import gym
#import roboschool




class Normalizer():
    def __init__(self, state_dim, action_dim):
        self.state_mean = np.zeros((1, state_dim))
        self.state_std = np.zeros((1, state_dim))

        self.action_mean = np.zeros((1, action_dim))
        self.action_std = np.zeros((1, action_dim))

        self.delta_state_mean = np.zeros((1, state_dim))
        self.delta_state_std = np.zeros((1, state_dim))

        self.norm_dict = {"state_mean" : self.state_mean, "state_std" : self.state_std, "act_mean" : self.action_mean, "act_std" : self.action_std, "delta_state_mean" : self.delta_state_mean, "delta_state_std" : self.delta_state_std}

        self.num_samples = 0

    def fit(self, states, actions, next_states):
        #states/acts should be given as (num_samples, state/act)
        #Fit works as a constant moving average: re-fitting data means updating the stats.
        # It's useful because the distribution of states can change changing the policy over time.


        delta_states = next_states - states

        self.state_mean[:,:] = np.mean(states, axis = 0)
        self.action_mean[:,:] = np.mean(actions, axis = 0)
        self.delta_state_mean[:,:] = np.mean(delta_states, axis = 0)

        self.state_std[:,:] = np.std(states, axis = 0)
        self.action_std[:,:] = np.std(actions, axis = 0)
        self.delta_state_std[:,:] = np.std(delta_states, axis = 0)

        new_samples = len(states)

        self.norm_dict["state_mean"] = self.norm_dict["state_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.state_mean* (new_samples/(self.num_samples + new_samples))
        self.norm_dict["act_mean"] = self.norm_dict["act_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.action_mean*(new_samples/(self.num_samples + new_samples))
        self.norm_dict["delta_state_mean"] = self.norm_dict["delta_state_mean"]*(1 - new_samples/(self.num_samples + new_samples)) + self.delta_state_mean*(new_samples/(self.num_samples + new_samples))

        self.norm_dict["state_std"] = self.norm_dict["state_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.state_std*(new_samples/(self.num_samples + new_samples)) + 1e-8
        self.norm_dict["act_std"] = self.norm_dict["act_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.action_std*(new_samples/(self.num_samples + new_samples)) + 1e-8
        self.norm_dict["delta_state_std"] = self.norm_dict["delta_state_std"]*(1 - new_samples/(self.num_samples + new_samples)) + self.delta_state_std*(new_samples/(self.num_samples + new_samples)) + 1e-8


        self.num_samples+=len(states)
        print(self.num_samples)

        return self.norm_dict





class DynNet(tf.keras.Model):
    def __init__(self, n_inp, n_out, hid_size):
        super(DynNet, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units = hid_size, activation = "relu")
        self.linear2 = tf.keras.layers.Dense(units = hid_size, activation = "relu")
        self.linear3 = tf.keras.layers.Dense(units = n_out)
        self.n_inp = n_inp
        self.initialize()

    def initialize(self):
        self.call(tf.convert_to_tensor(np.random.randn(1, self.n_inp)))
        for var in self.variables:
            if not "bias" in var.name:
                #print(var.shape)
                pass

    def call(self, x):
   #     x = tf.convert_to_tensor(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class NNDynamicsModel():
    def __init__(self,
                 state_dim,
                 action_dim,
                 hid_size,


                 normalization, #dict that has mean and std for each feature of state, acts, delta_states
                 batch_size,
                 iterations,
                 learning_rate
                 ):
        self.state_size = state_dim
        self.act_size = action_dim

        #create a network from the class above
        self.network = DynNet(n_inp = self.state_size + self.act_size, n_out = self.state_size, hid_size = hid_size)

        #store useful variables

        self.norm_dict = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate

        #create an optimizer
        self.opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    def train_step(self, x, y):

        with tf.GradientTape() as tape:
            out = self.network(tf.convert_to_tensor(x))
            loss = tf.reduce_mean(tf.losses.mean_squared_error(y, out))
            grads = tape.gradient(loss, self.network.variables)
            self.opt.apply_gradients(zip(grads, self.network.variables))
            return loss

    def fit(self, trajs, plot = 0):
        #fit: take a series of (s, a -> next_s)
        #normalize each one with the norm dict
        #run a backprop with tape with mse

        #trajs = dict states : (n_samples, state) dict acts : (n_samples, act), dict delta_states : (n_samples, state)

        states, acts, next_states = trajs["states"], trajs["acts"], trajs["next_states"]

        #obtain deltas
        delta_states = (next_states - states)
    #    print("states shape", states.shape)
    #    print("acts shape", acts.shape)
    #    print("delta states shape", delta_states.shape)


        state_mean = self.norm_dict["state_mean"]
        state_std = self.norm_dict["state_std"]
        act_mean = self.norm_dict["act_mean"]
        act_std = self.norm_dict["act_std"]
        delta_state_mean = self.norm_dict["delta_state_mean"]
        delta_state_std = self.norm_dict["delta_state_std"]


        #be sure state_mean = (1, state_dim)
    #    print("states mean shape", state_mean.shape, state_std.shape)
        states = (states - state_mean)/state_std

    #    print("acts mean shape", act_mean.shape, act_std.shape)
        acts = (acts - act_mean)/act_std

    #    print("delta_states mean shape", delta_state_mean.shape, delta_state_std.shape)
        delta_states = (delta_states - delta_state_mean)/delta_state_std

        #concatenate states and actions along axis 1 to create the real input
        #should obtain x = (num_samples, states + actions)

        inputs = np.concatenate((states, acts), axis = 1)

        #get train indices to shuffle them only once (probably faster than sampling batches?)

        train_indices = np.arange(states.shape[0]) # lenght = num_samples

        #train eval split

        val_indices = train_indices[-len(train_indices)//10:] #take last 10 percent so you sould always pick the new data
        train_indices = train_indices[: -len(train_indices)//10] #take first 90 percent

    #    print("train indices shape", train_indices.shape)
    #    print("eval indices shape", val_indices.shape)

        if plot:
            losses = []
        for ep in range(self.iterations):
            np.random.shuffle(train_indices)

            for batch in range((len(train_indices) // self.batch_size) + 1): # +1 to get the final batch, smaller than batch_size
                start = batch*self.batch_size
                indices_batch = train_indices[start : start + self.batch_size]

                #I get a list of indices. giving it to np array will give those elements back

                input_batch = inputs[indices_batch, :]
                output_batch = delta_states[indices_batch, :]

          #      input_batch+=tf.random_normal(shape=input_batch.shape, dtype=tf.float64)*0.0005
                loss = self.train_step(input_batch, output_batch)
                if plot:
                    losses.append(loss.numpy())

        if plot:
            plt.plot(losses)
            plt.show()

        val_loss = []
        for batch in range((len(val_indices) // self.batch_size) + 1): # +1 to get the final batch, smaller than batch_size
                start = batch*self.batch_size
                indices_batch = val_indices[start : start + self.batch_size]

                #I get a list of indices. giving it to np array will give those elements back

                input_batch = inputs[indices_batch, :]
                output_batch = delta_states[indices_batch, :]

                out = self.network(tf.convert_to_tensor(input_batch))
                loss = tf.reduce_mean(tf.losses.mean_squared_error(output_batch, out))
                val_loss.append(loss)
        #        print(loss)

        #print("Validation average loss", np.array(val_loss).mean())



    def predict(self, states, acts, next_states = False):
        #predict: take a series of (s,a)
        #normalize them with norm dict
        #go thoruhg the network
        #denormalize and give as output
        # Returns : unnormalized delta_states, if next_states = True predicts next states directly

        norm_states = (states - self.norm_dict["state_mean"])/self.norm_dict["state_std"]
        norm_acts = (acts - self.norm_dict["act_mean"])/self.norm_dict["act_std"]

        inp = tf.concat((norm_states, norm_acts), axis = 1) #concatenate along features axis
        out = self.network(inp)

    #    print("output shape", out.shape)
    #    print("out * std shape", out*self.norm_dict["delta_state_std"])
        #denormalize output
        out = out*self.norm_dict["delta_state_std"] + self.norm_dict["delta_state_mean"]

        if next_states: out+=states
        return out
        #CHECK SHAPES!!

    def unroll(self, state, actions, next_states = True):

        predicted_states = []
        actions = actions.reshape((-1, self.act_size))
        for act_i in range(len(actions)):
            action = actions[act_i].reshape((1,-1))
            next_state = self.predict(np.reshape(state, (1,-1)), action, next_states = True)
            predicted_states.append(next_state)
            state = next_state

        return predicted_states
