import math
def positional_encoding(seq):
    output=[]
    for pos in range(len(seq)):
        row=[]
        for i in range(len(seq[0])):
            value=math.sin(pos/(10000**(i/len(seq[0]))))
            row.append(value)
        output.append(row)
    return output

def add(A,B):
    output=[]
    for i in range(len(A)):
        row=[]
        for j in range(len(A[0])):
            row.append(A[i][j]+B[i][j])
        output.append(row)
    return output
def dot(a,b):
    return sum(a[i]*b[i] for i in range(len(a)))
def softmax(X):
    exp_X=[math.exp(i)for i in X]
    s=sum(exp_X)
    return [i/s for i in exp_X]
def attention(Q,K,V):
    K_T=[]
    for i in range(len(K[0])):
        row=[]
        for j in range(len(K)):
            row.append(K[j][i])
        K_T.append(row)
    scores=[]
    for i in range(len(Q)):
        row=[]
        for j in range(len(K_T)):
            value=dot(Q[i],K_T[j])/(math.sqrt(len(Q[0])))
            row.append(value)
        scores.append(row)
    weights=[softmax(row) for row in scores]
    out=[]
    for i in range(len(weights)):
        row=[0]*len(V[0])
        for j in range(len(weights[i])):
            for k in range(len(V[0])):
                row[k]+=weights[i][j]*V[j][k]
        out.append(row)
    return out

def multi_head_attention(X):

    head1=attention(X,X,X)
    head2=attention(X,X,X)
    output=[]
    for i in range(len(head1)):
        output.append(head1[i]+head2[i])
    return output
def transformer_encoder(x):
    mha=multi_head_attention(x)
    ff=[]
    for row in mha:
        ff.append([0.5* x for x in row])
    return ff
      
    







X=[[1, 0, 1], [0, 1, 1], [1, 1, 0]]
X_pe=positional_encoding(X)
X_added=add(X_pe,X)
output=transformer_encoder(X_added)
pred=output[-1]
print(pred)
