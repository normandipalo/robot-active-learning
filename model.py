import tensorflow as tf
import numpy as np
from utils import *



class BCModel(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_dim, hid_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        super(BCModel, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [state_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(LayerNorm(hidden_dim))
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(units = action_size))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)


    @tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x

    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
        self.xm = x.mean()
        self.xstd = x.std()
        x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        #ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
#        self.xm = tf.Variable(x.mean())
#        self.xstd = tf.Variable(x.std())
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            self.train_step(el, print_loss, verbose)

    @tf.function
    def train_step(self, el, print_loss = False, verbose = False):
        with tf.GradientTape() as tape:
            x, y = el
            y_pred = self.call(x)
            loss = self._loss(y_pred, y)
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)


class BCModelDropout(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_dim, hid_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        super(BCModelDropout, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [state_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(units = action_size))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)


    @tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x

    def error(self, x):
        outs = []
        for i in range(5):
            y = x
            for l in self._layers:
                y = l(y)
                y = tf.keras.layers.Dropout(0.2)(y, training = True)
            outs.append(y)
        return np.std(np.array(outs))

    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
        self.xm = x.mean()
        self.xstd = x.std()
        x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
#        self.xm = tf.Variable(x.mean())
#        self.xstd = tf.Variable(x.std())
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            self.train_step(el, print_loss, verbose)

    @tf.function
    def train_step(self, el, print_loss = False, verbose = False):
        with tf.GradientTape() as tape:
            x, y = el
            y_pred = self.call(x)
            loss = self._loss(y_pred, y)
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)




class ConvBCModel(tf.keras.Model):
    def __init__(self, im_size, action_size, hid_layers, k_sizes, filters, lr, set_seed = None):
        assert type(k_sizes) is list
        assert type(filters) is list

        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None

        super(ConvBCModel, self).__init__()
        self._layers = []
        for _ in range(len(filters)):
            self._layers.append(tf.keras.layers.Conv2D(kernel_size = k_sizes[_], filters = filters[_],
                                                        activation = "relu", strides = 2))
        #    self._layers.append(tf.keras.layers.BatchNormalization())

        self._layers.append(tf.keras.layers.Flatten())
        self._layers.append(tf.keras.layers.Dense(units = action_size))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)


    #@tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x

    #@tf.function
    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
    #    self.xm = x.mean()
    #    self.xstd = x.std()
    #    x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            with tf.GradientTape() as tape:
                x, y = el
                y_pred = self.call(x)
                loss = self._loss(y_pred, y)
            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(zip(grads, self.variables))
            if print_loss: print(loss)


class BCPlanner(tf.keras.Model):
    def __init__(self, bc_model, dyn_model, unc_model, action_size, w1_expl, w2_expl, w_den):
        super(BCPlanner, self).__init__()
        self.dyn_model = dyn_model
        self.unc_model = unc_model
        self.action_size = action_size
        self.bc_model = bc_model
        self.w1, self.w2, self.w = w1_expl, w2_expl, w_den

    def plan_step(self, state, minimize = True):
        x = np.array(state)
        if minimize:
            min_unc = 1e8
            best_action = np.zeros(self.action_size)
            for i in range(50):
                action = np.random.randn(5, self.action_size)
                tot_unc = 0
                for j in range(5):

                    next_state_pred = self.dyn_model.predict(x.reshape((1,-1)), action[j].reshape((1,-1)), True)
                    unc_on_fut = self.unc_model.error(next_state_pred)
                    tot_unc+=unc_on_fut.numpy()
                #print("Future Uncertainty", unc_on_fut)
                if tot_unc < min_unc:
                    min_unc = tot_unc
                    best_action = action[0]
            return best_action, min_unc
        else:
            max_unc = -1e6
            best_action = np.zeros(self.action_size)
            for i in range(50):
                action = np.random.randn(5, self.action_size)
                tot_unc = 0
                for j in range(5):
                    next_state_pred = self.dyn_model.predict(x.reshape((1,-1)), action[j].reshape((1,-1)), True)
                    unc_on_fut = self.unc_model.error(next_state_pred)
                    tot_unc+=unc_on_fut.numpy()
                #print("Future Uncertainty", unc_on_fut)
                if tot_unc > max_unc:
                    max_unc = tot_unc
                    best_action = action[0]
            return best_action


    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        self.bc_model.train(x, y, batch_size, epochs, print_loss, verbose)

    def call(self, x, minimize = True):
        action = self.bc_model.call(x)
        if minimize:
            correction, unc = self.plan_step(x, minimize)
        else:
            correction = self.plan_step(x, minimize)
        #print("action and correction", action, correction)
        if minimize:
            weight = unc/self.w
            weight = np.clip(weight, 0, 0.8)
            #print("unc", unc)
            coin = np.random.randint(0,5)
            if coin == 0:
                #return (1-weight)*action + weight*correction # + correction
                return self.w1*action + self.w2*correction # + correction
            else: return action
        else:
            return self.w1*action + self.w2*correction



class ConvHybridNet(tf.keras.Model):
    def __init__(self, im_size, action_size, n_channels, hid_layers, k_sizes, common_filters, dec_filters, lr, set_seed = None):
        assert type(k_sizes) is list
        assert type(dec_filters) is list
        assert type(common_filters) is list

        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None

        super(ConvHybridNet, self).__init__()
        self.im_size = im_size
        self._commonlayers = []
        for _ in range(len(common_filters)):
            self._commonlayers.append(tf.keras.layers.Conv2D(kernel_size = k_sizes[_], filters = common_filters[_],
                                                        activation = "relu", strides = 1, padding = "same"))
        #    self._layers.append(tf.keras.layers.BatchNormalization())

        self._actlayers = []
        self._actlayers.append(tf.keras.layers.Flatten())
        self._actlayers.append(tf.keras.layers.Dense(units = action_size))

        self._delayers = []
        for _ in reversed(range(len(dec_filters))):
            self._delayers.append(tf.keras.layers.Conv2D(kernel_size = k_sizes[_], filters = dec_filters[_],
                                                        activation = "relu", strides = 1, padding = "same"))
        self._delayers.append(tf.keras.layers.Conv2D(kernel_size = 3, filters = n_channels, activation = "relu",
                                                    strides = 1, padding = "same"))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)

    @tf.function
    def call(self, x):
        for l in self._commonlayers:
            x = l(x)
            x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)
        encoded = x


        for l in self._actlayers:
            encoded = l(encoded)

        actions = encoded
        #x = tf.reshape(x, (x.shape[0],
        """
        for i, l in enumerate(self._delayers):

            x = l(x)
            x = tf.keras.layers.UpSampling2D()(x) if i != len(self._delayers) - 1 else x

        pad = x.shape[1] - self.im_size
        """
        #return actions, x[:,pad//2:x.shape[1] - pad//2, pad//2:x.shape[1]-pad//2,:]
        return actions, tf.zeros((self.im_size, self.im_size, 4))

    @tf.function
    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
    #    self.xm = x.mean()
    #    self.xstd = x.std()
    #    x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds

    @tf.function
    def error(self, x):
        a, x_hat = self.call(x)
        return tf.reduce_mean(tf.losses.mean_squared_error(x, x_hat))

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            self.train_step(el, print_loss, verbose)

    @tf.function
    def train_step(self, el, print_loss = False, verbose = False):
        with tf.GradientTape() as tape:
            x, y = el
            y_pred, x_pred = self.call(x)
            loss = self._loss(y_pred, y)
            #loss += self._loss(x_pred, x)
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)

class ConvHybridNet2(tf.keras.Model):
    def __init__(self, im_size, action_size, n_channels, hid_layers, k_sizes, common_filters, dec_units, dec_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None

        super(ConvHybridNet2, self).__init__()
        self.im_size = im_size
        self.common_filters = common_filters
        self._commonlayers = []
        for _ in range(len(common_filters)):
            self._commonlayers.append(tf.keras.layers.Conv2D(kernel_size = k_sizes[_], filters = common_filters[_],
                                                        activation = "relu", strides = 1, padding = "same"))
        #    self._layers.append(tf.keras.layers.BatchNormalization())

        self._actlayers = []
        self._actlayers.append(tf.keras.layers.Flatten())
        self._actlayers.append(tf.keras.layers.Dense(units = action_size))

        self._delayers = []

        #Compute the flattened shape.
        self.flattened_shape = (self.im_size//(2**len(self.common_filters))) * (self.im_size//(2**len(self.common_filters))) * self.common_filters[-1]
        for _ in reversed(range(dec_layers)):
            self._delayers.append(tf.keras.layers.Dense(units = dec_units,
                                                        activation = "relu"))
        self._delayers.append(tf.keras.layers.Dense(units = self.flattened_shape,
                                                    activation = "relu"))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)

    @tf.function
    def call(self, x):
        for l in self._commonlayers:
            x = l(x)
            x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)
        encoded = x


        for l in self._actlayers:
            encoded = l(encoded)

        actions = encoded
        #x = tf.reshape(x, (x.shape[0],
        """
        for i, l in enumerate(self._delayers):

            x = l(x)
            x = tf.keras.layers.UpSampling2D()(x) if i != len(self._delayers) - 1 else x

        pad = x.shape[1] - self.im_size
        """
        #return actions, x[:,pad//2:x.shape[1] - pad//2, pad//2:x.shape[1]-pad//2,:]
        return actions, tf.zeros((self.im_size, self.im_size, 4))

    @tf.function
    def call_complete(self, x):
        for l in self._commonlayers:
            x = l(x)
            x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)
        embedded = x


        for l in self._actlayers:
            embedded = l(embedded)

        actions = embedded


        encoded = tf.keras.layers.Flatten()(x)

        for l in self._delayers:
            encoded = l(encoded)
        enc_error = tf.reduce_mean(tf.losses.mean_squared_error(tf.stop_gradient(tf.keras.layers.Flatten()(x)), encoded))
        return actions, enc_error

    @tf.function
    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
    #    self.xm = x.mean()
    #    self.xstd = x.std()
    #    x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds

    @tf.function
    def error(self, x):
        for l in self._commonlayers:
            x = l(x)
            x = tf.keras.layers.MaxPool2D(pool_size = (2,2))(x)
        encoded = x
        encoded = tf.keras.layers.Flatten()(encoded)

        for l in self._delayers:
            encoded = l(encoded)

        return tf.reduce_mean(tf.losses.mean_squared_error(tf.stop_gradient(tf.keras.layers.Flatten()(x)), encoded))

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            self.train_step(el, print_loss, verbose)

    @tf.function
    def train_step(self, el, print_loss = False, verbose = False):
        with tf.GradientTape() as tape:
            x, y = el
            y_pred, pred_err = self.call_complete(x)
            loss = self._loss(y_pred, y)
            loss += pred_err
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)

class BCModelDropout(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_dim, hid_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        super(BCModelDropout, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [state_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(units = action_size))
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)


    @tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x

    def error(self, x):
        outs = []
        for i in range(5):
            y = x
            for l in self._layers:
                y = l(y)
                y = tf.keras.layers.Dropout(0.2)(y, training = True)
            outs.append(y)
        return np.std(np.array(outs))

    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)

    def _create_ds(self, x, y, batch_size, epochs):
        self.xm = x.mean()
        self.xstd = x.std()
        x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds

    def train(self, x, y, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
#        self.xm = tf.Variable(x.mean())
#        self.xstd = tf.Variable(x.std())
        ds = self._create_ds(x, y, batch_size, epochs)
        for i, el in enumerate(ds):
            if verbose:
                if i%1000==0: print("Element ", i)
            self.train_step(el, print_loss, verbose)

    @tf.function
    def train_step(self, el, print_loss = False, verbose = False):
        with tf.GradientTape() as tape:
            x, y = el
            y_pred = self.call(x)
            loss = self._loss(y_pred, y)
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)
