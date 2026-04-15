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