
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error


X=np.load('X.npy')
Y=np.load('Y.npy')


m=X.shape[0]
	
#feature normaliaztion
Xn=np.zeros( X.shape )
for i in range ( X.shape[1] ) :
	s=X[:,i].max()-X[:,i].min()
	u=np.mean(X[:,i])
	Xn[:,i]=(X[:,i]-u )/s

#adding bias	
X1=np.zeros((m,X.shape[1]+1) )
X1[:,0]=np.ones((m,1) )[:,0]	
X1[ :,1:X1.shape[1] ]=Xn


"""
# adding quadradic features
X=np.load('X.npy')
Y=np.load('Y.npy')
Xsq=np.multiply(X,X)
Xtq=np.power(X,3)
X=np.append(X,Xsq,axis=1)
"""
m1=X.shape[0]
"""
valErr=[]
trainErr=[]
for j in range(50,98,1):
	Xm=X[0:(j+1),:]
	Ym=Y[0:(j+1),:]
	print Xm.shape
	m=Xm.shape[0]
	
	#feature normaliaztion
	Xn=np.zeros( Xm.shape )
	for i in range ( Xm.shape[1] ) :
		s=Xm[:,i].max()-Xm[:,i].min()
		u=np.mean(Xm[:,i])
		Xn[:,i]=(Xm[:,i]-u )/s

	#adding bias	
	X1=np.zeros((m,Xm.shape[1]+1) )
	X1[:,0]=np.ones((m,1) )[:,0]	
	X1[ :,1:X1.shape[1] ]=Xm

	# Split data to train and test
	Xt,Xval,Yt ,Yval =train_test_split(X1,Ym,test_size=0.1, random_state=42)


	# Neural Net setup and Train 
	#best=(activation='relu',hidden_layer_sizes=(30,30,20,17),max_iter=1000,solver='lbfgs',alpha=50.11,learning_rate_init=0.0001)
	#mea=4
	NN=MLPRegressor(activation='relu',hidden_layer_sizes=(15,15,15,7),max_iter=1500,solver='lbfgs',alpha=250.11,learning_rate_init=0.00001)
	NN.fit(Xt,Yt)
	pred=NN.predict(Xval)
	predTrain=NN.predict(Xt)
		

	valErr.append(mean_absolute_error(Yval,pred) )
	trainErr.append(mean_absolute_error(Yt,predTrain) )
	
	

Train_Erorr=plt.plot(range(50,98),trainErr,'r-',label='Training Error')
Val_Error =plt.plot(range(50,98),valErr,'b-',label='Test Error')
print valErr[-1]
plt.legend()
plt.xlabel('Traing Examples')
plt.ylabel('mean absulote error')
plt.show()
"""
c=0
err=[]

for k in range(10):
	test_arr=range(c,c+10)
	c+=10
	Xt=Xn
	Yt=Y
	Xval=np.zeros((10,Xn.shape[1]))
	Yval=np.zeros((10,2))
	for i in range(len(test_arr)):
		Xval[i]=Xn[test_arr[i]-i]
		Yval[i]=Y[test_arr[i]-i]
		Xt=np.delete(Xt,test_arr[i]-i,0)
		Yt=np.delete(Yt,test_arr[i]-i,0)
##################(activation='relu',hidden_layer_sizes=(60,60,60,60),max_iter=1500,solver='lbfgs',alpha=.011,learning_rate_init=0.0001)
	NN=MLPRegressor(activation='relu',hidden_layer_sizes=(60,60,60,60),max_iter=1500,solver='lbfgs',alpha=.011,learning_rate_init=0.0001)
	NN.fit(Xt,Yt)
	pred=NN.predict(Xval)
	predTrain=NN.predict(Xt)
	err.append(mean_absolute_error(pred,Yval))
	print "K=" , k+1
	print "Test error :" , mean_absolute_error(pred,Yval)
	print  "Train error :",mean_absolute_error(predTrain,Yt)


print "mean 10 flod error",(sum(err)/len(err))
