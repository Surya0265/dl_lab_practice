# ----------- DATASET -----------
image = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,0]
]

# New pattern for prediction
test_image = [
    [0,1,0,0],
    [1,1,1,0],
    [0,1,0,0],
    [0,0,0,0]
]

# ----------- CNN MODEL (KERNEL) -----------
kernel = [
    [0.1, 0.1, 0.1],
    [0.1, 0.5, 0.1],
    [0.1, 0.1, 0.1]
]

# ----------- CONVOLUTION FUNCTION -----------
def convolve(img, ker):
    feature_map = []
    for i in range(2):  # 4-3+1 = 2
        row = []
        for j in range(2):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    s += img[i+ki][j+kj] * ker[ki][kj]
            row.append(s)
        feature_map.append(row)
    return feature_map

# ----------- TRAINING FUNCTION -----------
def train(lr):
    global kernel
    target = 1  # cross exists

    print("\nTraining with learning rate =", lr)

    for epoch in range(10):
        feature_map = convolve(image, kernel)

        # Compute output (sum of feature map)
        output = 0
        for r in feature_map:
            for v in r:
                output += v

        error = target - output

        # Update kernel
        for i in range(3):
            for j in range(3):
                grad = 0
                for x in range(2):
                    for y in range(2):
                        grad += image[x+i][y+j]
                kernel[i][j] += lr * error * grad

        print("Epoch:", epoch, "Output:", output)

    print("Updated Kernel:", kernel)

# ----------- TRAIN WITH DIFFERENT LEARNING RATES -----------
train(0.01)
train(0.1)

# ----------- PREDICTION -----------
feature_map = convolve(test_image, kernel)

output = 0
for r in feature_map:
    for v in r:
        output += v

prediction = 1 if output > 1 else 0

print("\nPrediction (1=Cross Detected, 0=Not Detected):", prediction)

# ----------- ACTIVATION MAP -----------
print("\nActivation Map:")
for row in feature_map:
    print(row)