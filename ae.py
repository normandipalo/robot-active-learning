import tensorflow as tf
import numpy as np

class AE(tf.keras.Model):
    def __init__(self, in_size, hidden_dim, hid_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        super(AE, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [in_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(in_size))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)
        
        
    #@tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x
    
    #@tf.function
    def error(self, x):
        y = self.call(x)
        return tf.reduce_sum(tf.losses.mean_squared_error(y,x))
    
    
    #@tf.function
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
    
    def train(self, x, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, batch_size, epochs)    
        for i,el in enumerate(ds):
            if verbose: 
                if i%1000==0: print("Element ", i)
            with tf.GradientTape() as tape:
                x = el
                y_pred = self.call(x)
                loss = self._loss(y_pred, x)
            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(zip(grads, self.variables))
            if print_loss: print(loss)
                            
                           

class DAE(tf.keras.Model):
    def __init__(self, in_size, hidden_dim, hid_layers, lr, set_seed = None):
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        super(DAE, self).__init__(self)
        self._layers = []
        self._layers.append(tf.keras.layers.Dense(input_shape = [in_size],  units = hidden_dim, activation = "relu"))
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(in_size))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)
        
        
    #@tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x
    
    #@tf.function
    def error(self, x):
        y = self.call(x)
        return tf.reduce_sum(tf.losses.mean_squared_error(y,x))
    
    
    #@tf.function
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
    
    def train(self, x, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, batch_size, epochs)    
        for i,el in enumerate(ds):
            if verbose: 
                if i%1000==0: print("Element ", i)
            with tf.GradientTape() as tape:
                x = el
                x_n = x + np.random.randn(*el.shape)*0.35
                y_pred = self.call(x_n)
                loss = self._loss(y_pred, x)
            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(zip(grads, self.variables))
            if print_loss: print(loss)
                            
                           
                           
                           
class RandomNetwork(tf.keras.Model):
    def __init__(self, out_size, hidden_dim, hid_layers, lr, set_seed = None):
        
        if set_seed:
            tf.random.set_seed(set_seed)
            self.set_seed = set_seed
        else:
            self.set_seed = None
        
        super(RandomNetwork, self).__init__()
        self.r_layers = []
        self._layers = []
        
        for i in range(4):
            self.r_layers.append(tf.keras.layers.Dense(units = 512, activation = "tanh"))
        self.r_layers.append(tf.keras.layers.Dense(units = out_size))
        
        for l in self.r_layers:
            for v in l.variables:
                v = tf.Variable(v, trainable = False)
        
        for i in range(hid_layers):
            self._layers.append(tf.keras.layers.Dense(units = hidden_dim, activation = "relu"))
        self._layers.append(tf.keras.layers.Dense(units = out_size))
        
        self.opt = tf.keras.optimizers.Adam(learning_rate = lr)
        
    #@tf.function
    def call(self, x):
        for l in self._layers:
            x = l(x)
        return x
        
    def call_rand(self, x):
        for l in self.r_layers:
            x = l(x)
        return x
    
    #@tf.function
    def error(self, x):
        y = self.call(x)
        y_r = self.call_rand(x)
        return tf.reduce_sum(tf.losses.mean_squared_error(y,y_r))
    
    
    #@tf.function
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
    
    def train(self, x, batch_size, epochs, print_loss = False, verbose = False):
        if self.set_seed:
            tf.random.set_seed(self.set_seed)
        ds = self._create_ds(x, batch_size, epochs)    
        for i,el in enumerate(ds):
            if verbose: 
                if i%1000==0: print("Element ", i)
            with tf.GradientTape() as tape:
                x = el
                y_pred = self.call(x)
                y_rand = self.call_rand(x)
                loss = self._loss(y_pred, tf.stop_gradient(y_rand))
                
            grads = tape.gradient(loss, self.variables)
            self.opt.apply_gradients(zip(grads, self.variables))
            if print_loss: print(loss)
