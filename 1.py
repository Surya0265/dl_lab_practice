import matplotlib.pyplot as plt

X=[500, 800, 1000, 1200, 1500]
Y=[150, 220, 300, 360, 450 ]

epochs=1000
w=0.0001
b=0.1
lr=0.0000001
for epoch in range(epochs+1):
    total_loss=0
    dw=0
    db=0
    for i in range(len(X)):
        y=w*X[i]+b
        error=(Y[i]-y)
        total_loss+=error**2
        dw+=-2*error*X[i]
        db+=-2*error
    loss=total_loss/len(X)
    dw=dw/len(X)
    db=db/len(X)
    w-=lr*dw
    b-=lr*db
    if epoch%100==0:
        print("epoch:",epoch,"loss:",round(loss,4))
plt.scatter(X,Y,label="Data")
x_line=[i for i in range(500,1600,10)]
y_line=[w*x_line[i]+b for i in range(len(x_line))]
plt.plot(x_line,y_line,label="Reg")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()

   
