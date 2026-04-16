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


def max_pool(feature):
    output=[]
    for i in range(0,3.2):
        row=[]
        for j in range(0,3,2):
            m=feature[i][j]
            for x in range(2):
                for y in range(2):
                    if i+x<3 and j+x<3:
                        m=max(m,feature[i+x][j+y])
            row.append(m)
        output.append(row)
