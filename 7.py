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



          
