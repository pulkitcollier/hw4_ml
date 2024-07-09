import time
import numpy as np
import scipy.io as sio

start_time = time.time()
dataset = sio.loadmat('hw4data.mat')

labels = dataset.get('labels')
data = dataset.get('data')

data = data/[20,1,20]

d = data.shape[1]
n = data.shape[0]
beta_not = 0
beta = np.array(np.zeros(d))
beta = np.append(beta,beta_not)
data = np.append(data,np.array(np.ones((n,1))),1)
eta = 1
objective = (np.sum(np.log(1+np.exp(np.dot(data,beta))))-np.dot((np.dot(data,beta)),labels))/n
gradient = ((np.dot(((1/(1+np.exp(-np.dot(data,beta))))-labels.T),data))/n)[0]
cnt = 0
while(objective>0.65064):
    gradient = ((np.dot(((1/(1+np.exp(-np.dot(data,beta))))-labels.T),data))/n)[0]
    L = (np.sum(np.log(1+np.exp(np.dot(data,(beta-(gradient*eta))))))-np.dot((np.dot(data,(beta-(gradient*eta)))),labels))/n
    R = objective - ((eta*(np.sum(np.square(gradient))))/2)
    while(L>R):
        eta = eta/2
        L = (np.sum(np.log(1+np.exp(np.dot(data,(beta-(gradient*eta))))))-np.dot((np.dot(data,(beta-(gradient*eta)))),labels))/n 
        R = objective - ((eta*(np.sum(np.square(gradient))))/2)
        print(eta)
    beta = beta - (eta*gradient)
    print(objective[0])
    objective = (np.sum(np.log(1+np.exp(np.dot(data,beta))))-np.dot((np.dot(data,beta)),labels))/n
    cnt +=1
print(cnt)
print(time.time()-start_time)
