import tensorflow as tf
import numpy as np

# -------- 1. Dataset (16-dim vectors) --------
X = np.array([
 [1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1],
 [0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0],
 [1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0],
 [1,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1]
], float)

# -------- Sampling (Reparameterization) --------
def sample(args):
    z_mean, z_log = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)))
    return z_mean + tf.exp(0.5*z_log) * eps

# -------- Build VAE --------
def build_vae(lr):
    inp = tf.keras.Input(shape=(16,))
    
    # Encoder
    h = tf.keras.layers.Dense(8, activation='relu')(inp)
    z_mean = tf.keras.layers.Dense(2)(h)
    z_log  = tf.keras.layers.Dense(2)(h)
    
    z = tf.keras.layers.Lambda(sample)([z_mean, z_log])
    
    # Decoder
    d = tf.keras.layers.Dense(8, activation='relu')(z)
    out = tf.keras.layers.Dense(16, activation='sigmoid')(d)
    
    model = tf.keras.Model(inp, out)
    
    # Loss = Reconstruction + KL
    recon = tf.reduce_mean((inp - out)**2)
    kl = -0.5 * tf.reduce_mean(1 + z_log - tf.square(z_mean) - tf.exp(z_log))
    
    model.add_loss(recon + kl)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    return model

# -------- Train & Generate --------
def run(lr):
    model = build_vae(lr)
    
    model.fit(X, X, epochs=10, verbose=0)
    
    # Generate new sample
    z = np.random.randn(1,2)
    gen = model.predict(z)
    
    print("\nLR:", lr)
    print("Generated:\n", np.round(gen))
    
# -------- Run for both LR --------
run(0.001)
run(0.01)