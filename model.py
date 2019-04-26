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
        
        
    #@tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x
    
    #@tf.function
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
            
            
