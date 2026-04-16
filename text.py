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



import math

def positional_encoding(X):
     out=[]
     for i in range(len(X)):
        out.append(math.sin(X[i]))
     return out
def softmax(X):
    exp_X=[math.exp(x) for x in X]
    s=sum(exp_X)
    return [i/s for i in exp_X]

def attention(X):
    scores=[]
    for i in range(len(X)):
        row=[]
        for j in range(len(X)):
            row.append(X[i]*X[j]/math.sqrt(len(X)))
        scores.append(row)
    weights=[softmax(row) for row in scores]
    out=[]
    for i in range(len(weights)):
        value=0
        for j in range(len(weights[i])):
            value+=weights[i][j]*X[j]
        out.append(value)
    return out
        
        
      

      
def train(lr):
    w=0.5
    X=[1, 0, 1, 1, 0]
    target=[0,1,1,0,0]
    X_pe=positional_encoding(X)
    X=[X[i]+X_pe[i] for i in range(len(X))]

    for t in range(10):
         out=attention(X)
         for i in range(len(out)):
            y=sigmoid(out[i]*w)
            error=target[i]-y
            d=-2*error*y*(1-y)
            w-=lr*d*out[i]
         print("loss:",round(error,4))
         out=attention(X)
         pred=sigmoid(w*out[-1])
         print(pred)














train(0.001)
import math
import random


real_data = [
    [1,0,1,0],
    [0,1,0,1]
]

G = [0.5, -0.5, 0.5, -0.5]
D = [0.3, 0.3, 0.3, 0.3]
b = 0.1
def sigmoid(x):
    return 1/(1+(math.exp(-x)))


lr=0.001
epochs=10
for e in range(epochs):
     total_loss=0
     for real in real_data:
          d_real=sigmoid(sum(D[i]*real[i] for i in range(4))+b)
          noise=[random.random() for i in range(4)]
          fake=[sigmoid(G[i]*noise[i]) for i in range(4)]
          d_fake=sigmoid(sum(D[i]*fake[i] for i in range(4))+b)
          loss=(1-(d_real**2))*(d_fake**2)
          total_loss+=loss
          for i in range(len(D)):
            D[i]+=lr*((1-real[i])*real[i]-d_fake*fake[i])
     for i in range(2):
        noise=[random.random() for i in range(4)]
        fake=[sigmoid(G[i]*noise[i]) for i in range(4)]
        d_fake=sigmoid(sum(D[i]*fake[i] for i in range(4))+b)
        G[i]+=(1-d_fake)*noise[i]
     print(loss)


print([sigmoid(G[i]*noise[i])for i in range(4)])
import math

# Sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# -------- Dataset (image as sequence) --------
# Each row is a timestep
X = [
    [1,1,1],   # t1
    [0,1,0],   # t2
    [1,1,1]    # t3
]

# Target class (example: 1)
target = 1

# -------- Initialize weights --------
Wx = [0.5, 0.5, 0.5]   # input → hidden
Wh = 0.4               # hidden → hidden
Wy = 0.6               # hidden → output

bh = 0.1
by = 0.1

# -------- Training --------
def train(lr):
    global Wx, Wh, Wy, bh, by
    
    print("\nLearning rate:", lr)
    
    for epoch in range(10):
        h = 0   # initial hidden state
        states = []
        
        # -------- Forward pass --------
        for t in range(len(X)):
            x = X[t]
            
            # input contribution
            x_sum = sum(x[i]*Wx[i] for i in range(3))
            
            # hidden update
            h = sigmoid(x_sum + Wh*h + bh)
            states.append(h)
        
        # output
        y = sigmoid(Wy*h + by)
        
        # loss
        error = target - y
        loss = error**2
        
        # -------- Backprop (simple) --------
        d_out = -2*(target - y)*y*(1-y)
        
        # update output weights
        Wy -= lr * d_out * h
        by -= lr * d_out
        
        # update hidden weights (simplified)
        for t in range(len(states)):
            Wx[0] -= lr * d_out * states[t]
            Wx[1] -= lr * d_out * states[t]
            Wx[2] -= lr * d_out * states[t]
        
        Wh -= lr * d_out * h
        
        print("Epoch", epoch, "Loss =", round(loss, 4))
    
    print("Prediction:", round(y))


# -------- Run for both learning rates --------
train(0.01)
train(0.1)

# Input image (3x3)
image = [
    [10,10,10],
    [0,0,0],
    [10,10,10]
]

# Initial kernel (horizontal edge detector)
kernel = [
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
]

# Convolution function
def convolve(img, ker):
    result = 0
    for i in range(3):
        for j in range(3):
            result += img[i][j] * ker[i][j]
    return result

# Training function
def train(lr):
    global kernel
    target = 1

    print("\nTraining with learning rate =", lr)

    for epoch in range(10):
        output = convolve(image, kernel)
        error = target - output

        # Update kernel weights
        for i in range(3):
            for j in range(3):
                kernel[i][j] += lr * error * image[i][j]

        print("Epoch:", epoch, "Output:", output)

    print("Final Kernel:", kernel)


# Run training for different learning rates
train(0.01)
train(0.5)

# Input pattern (4x4)
image = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,0]
]

# New pattern for testing
test_image = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,0]
]

# Initialize 3x3 kernel (random values)
kernel = [
    [0.1, 0.1, 0.1],
    [0.1, 0.5, 0.1],
    [0.1, 0.1, 0.1]
]

# Convolution (2x2 output for 4x4 input & 3x3 kernel)
def convolve(img, ker):
    output = []
    for i in range(2):  # 4-3+1 = 2
        row = []
        for j in range(2):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    s += img[i+ki][j+kj] * ker[ki][kj]
            row.append(s)
        output.append(row)
    return output

# Simple prediction (sum of feature map)
def predict(feature_map):
    total = 0
    for row in feature_map:
        for val in row:
            total += val
    return 1 if total > 1 else 0   # threshold

# Training function
def train(lr):
    global kernel
    target = 1   # cross pattern exists

    print("\nTraining with learning rate =", lr)

    for epoch in range(10):
        feature_map = convolve(image, kernel)

        # calculate output
        total = 0
        for r in feature_map:
            for v in r:
                total += v

        error = target - total

        # update kernel
        for i in range(3):
            for j in range(3):
                grad = 0
                for x in range(2):
                    for y in range(2):
                        grad += image[x+i][y+j]
                kernel[i][j] += lr * error * grad

        print("Epoch:", epoch, "Output:", total)

    print("Final Kernel:", kernel)


# Train with two learning rates
train(0.01)
train(0.1)

# Prediction on new pattern
feature_map = convolve(test_image, kernel)
prediction = predict(feature_map)

print("\nPrediction (1 = Cross detected, 0 = Not detected):", prediction)

# Activation map visualization
print("\nActivation Map:")
for row in feature_map:
    print(row)