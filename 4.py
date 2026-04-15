import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Dataset
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0,1,1,0]

W = [[0.5, -0.5],
     [0.3,  0.8]]

V = [0.7, -0.6]

b_hidden = [0.1, 0.1]
b_out = 0.1

eta = 0.01
epochs = 10

for e in range(epochs):
    total_loss = 0   # ✅ initialize
    
    for i in range(4):
        x1, x2 = X[i]
        t = Y[i]
        
        # Forward
        h1 = sigmoid(W[0][0]*x1 + W[0][1]*x2 + b_hidden[0])
        h2 = sigmoid(W[1][0]*x1 + W[1][1]*x2 + b_hidden[1])
        
        y = sigmoid(h1*V[0] + h2*V[1] + b_out)
        
        error = t - y
        total_loss += error**2   # ✅ accumulate
        
        # Backprop
        d_out = -2*(t-y)*y*(1-y)
        d_h1 = d_out * V[0] * h1*(1-h1)
        d_h2 = d_out * V[1] * h2*(1-h2)
        
        # Update output
        V[0] -= eta * d_out * h1
        V[1] -= eta * d_out * h2
        b_out -= eta * d_out
        
        # Update hidden
        W[0][0] -= eta * d_h1 * x1
        W[0][1] -= eta * d_h1 * x2
        W[1][0] -= eta * d_h2 * x1
        W[1][1] -= eta * d_h2 * x2
        
        b_hidden[0] -= eta * d_h1
        b_hidden[1] -= eta * d_h2
    
    print("Epoch", e, "Loss =", round(total_loss, 4))

print("\nFinal Weights:", W)