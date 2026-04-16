# -------- DATA (RESHAPED 5x5) --------
zero = [
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1]
]

one = [
    [0,0,1,0,0],
    [0,1,1,0,0],
    [1,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1]
]

# New test pattern
test = one

# -------- KERNEL (3x3) --------
kernel = [
    [0.1,0.1,0.1],
    [0.1,0.5,0.1],
    [0.1,0.1,0.1]
]

# -------- CONVOLUTION --------
def conv(img, ker):
    out = []
    for i in range(3):   # 5-3+1 = 3
        row = []
        for j in range(3):
            s = 0
            for ki in range(3):
                for kj in range(3):
                    s += img[i+ki][j+kj] * ker[ki][kj]
            row.append(s)
        out.append(row)
    return out

# -------- MAX POOLING (2x2 → 1 value per region) --------
def maxpool(fm):
    pooled = []
    for i in range(0,3,2):
        row = []
        for j in range(0,3,2):
            m = fm[i][j]
            for x in range(2):
                for y in range(2):
                    if i+x < 3 and j+y < 3:
                        m = max(m, fm[i+x][j+y])
            row.append(m)
        pooled.append(row)
    return pooled

# -------- TRAIN --------
def train(img, target, lr):
    global kernel
    for epoch in range(10):
        fm = conv(img, kernel)
        pooled = maxpool(fm)

        # output = sum of pooled
        out = sum(sum(r) for r in pooled)
        error = target - out

        # update kernel
        for i in range(3):
            for j in range(3):
                kernel[i][j] += lr * error

        print("Epoch", epoch, "Output:", out)

# -------- TRAINING --------
print("Training for Digit 0")
train(zero, 0, 0.01)

print("\nTraining for Digit 1")
train(one, 1, 0.1)

# -------- PREDICTION --------
fm = conv(test, kernel)
pooled = maxpool(fm)

out = sum(sum(r) for r in pooled)
prediction = 1 if out > 1 else 0

print("\nPrediction (0 or 1):", prediction)

# -------- FEATURE MAP --------
print("\nFeature Map:")
for r in fm:
    print(r)

print("\nPooled Map:")
for r in pooled:
    print(r)