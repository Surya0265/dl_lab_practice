# ----------- DATASET -----------
img = [
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
def conv(img,kernel):
    feature_map=[]
    for i in range(2):
        row=[]
        for j in range(2):
            s=0
            for ki in range(3):
                for kj in range(3):
                    s+=img[i+ki][j+kj]*kernel[ki][kj]
            row.append(s)
        feature_map.append(row)
    return feature_map



def train(lr):
        global kernel
        print("learning rate:", lr)
        for i in range(10):
            feature_map=conv(img,kernel)
            output=0
            for feature in feature_map:
                for v in feature:
                    output+=v
            error=1-output
            for i in range(3):
                for j in range(3):
                    grad=0
                    for x in range(2):
                        for y in range(2):
                            grad+=img[i+x][j+y]
                    kernel[i][j]+=lr*error*grad

train(0.001)
print(kernel)
            

