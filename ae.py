import tensorflow as tf
import numpy as np

class AE(tf.keras.Model):
    def __init__(self, in_size, hidden_dim, hid_layers, lr):
        super(AE, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [in_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(in_size))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)
        
        
    @tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x
    
    @tf.function
    def error(self, x):
        y = self.call(x)
        return tf.reduce_sum(tf.losses.mean_squared_error(y,x))
    
    
    @tf.function
    def _loss(self, x, y):
        return tf.reduce_mean(tf.losses.mean_squared_error(y, x)) #.mean_squared_error(y, x)
    
    def _create_ds(self, x, batch_size, epochs):
        self.xm = x.mean()
        self.xstd = x.std()
        x = (x-self.xm)/self.xstd
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.shuffle(x.shape[0])
        ds = ds.repeat(epochs)
        ds = ds.batch(batch_size)
        return ds
    
    def train(self, x, batch_size, epochs):
        ds = self._create_ds(x, batch_size, epochs)    
        for el in ds:
            with tf.GradientTape() as tape:
                x = el
                y_pred = self.call(x)
                loss = self._loss(y_pred, x)
            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(zip(grads, self.variables))
            print(loss)
                            
                           