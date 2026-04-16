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

def train(lr):
    s=0
    global kernel
    for i in range(3):
        for j in range(3):
            s+=kernel[i][j]*img[i][j]
    out=s
    error=1-out

    for i in range(3):
        for j in range(3):
            kernel[i][j]+=lr*error*img[i][j]
    return s
    
s=train(0.001)
print(kernel)
print(s)



