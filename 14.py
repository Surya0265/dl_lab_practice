# -------- DATASET (3x3 IMAGE) --------
img = [
    [10,10,10],
    [0,0,0],
    [10,10,10]
]

# -------- KERNEL (HORIZONTAL EDGE) --------
kernel = [
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
]

# -------- CONVOLUTION --------
def conv(img, kernel):
    s = 0
    for i in range(3):
        for j in range(3):
            s += img[i][j] * kernel[i][j]
    return s

# -------- TRAINING --------
def train(lr):
    global kernel
    print("\nLearning rate:", lr)

    for epoch in range(10):
        output = conv(img, kernel)
        error = 1 - output   # target = 1

        # update kernel
        for i in range(3):
            for j in range(3):
                kernel[i][j] += lr * error * img[i][j]

        print("Epoch", epoch, "Output:", output)

    print("Updated Kernel:", kernel)

# -------- TRAIN WITH DIFFERENT LEARNING RATES --------
train(0.01)
train(0.5)

# -------- FEATURE MAP (OUTPUT) --------
feature_map = conv(img, kernel)

print("\nFeature Map (Edge Strength):", feature_map)

# -------- INTERPRETATION --------
if feature_map > 0:
    print("Edge Detected")
elif feature_map < 0:
    print("Opposite Edge Detected")
else:
    print("No Edge Detected")