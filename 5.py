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