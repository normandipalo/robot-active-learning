import tensorflow as tf
import numpy as np

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
        for i, l in enumerate(self._delayers):

            x = l(x)
            x = tf.keras.layers.UpSampling2D()(x) if i != len(self._delayers) - 1 else x

        pad = x.shape[1] - self.im_size
        return actions, x[:,pad//2:x.shape[1] - pad//2, pad//2:x.shape[1]-pad//2,:]

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
            loss += self._loss(x_pred, x)
        grads = tape.gradient(loss, self.variables)
        self.opt.apply_gradients(zip(grads, self.variables))
        if print_loss: print(loss)
