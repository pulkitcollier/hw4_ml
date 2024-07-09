import numpy as np
import scipy.io as sio


dataset = sio.loadmat('hw4data.mat')

labels = dataset.get('labels')
data = dataset.get('data')


d = data.shape[1]
n = data.shape[0]
n_training = int(0.8*n)
n_ho = n-n_training
ho_data = data[n_training:]
data = data[:n_training]
ho_labels = labels[n_training:]
labels =  labels[:n_training]
predicted_labels = np.zeros(ho_labels.shape[0])
beta_not = 0
beta = np.array(np.zeros(d))
beta = np.append(beta,beta_not)
data = np.append(data,np.array(np.ones((n_training,1))),1)
ho_data = np.append(ho_data,np.array(np.ones((n_ho,1))),1)
eta = 1
objective = (np.sum(np.log(1+np.exp(np.dot(data,beta))))-np.dot((np.dot(data,beta)),labels))/n_training
gradient = ((np.dot(((1/(1+np.exp(-np.dot(data,beta))))-labels.T),data))/n_training)[0]
cnt = 0
p = 2
ho_er_best = 1
ho_er = 0
while(1):
    gradient = ((np.dot(((1/(1+np.exp(-np.dot(data,beta))))-labels.T),data))/n_training)[0]
    L = (np.sum(np.log(1+np.exp(np.dot(data,(beta-(gradient*eta))))))-np.dot((np.dot(data,(beta-(gradient*eta)))),labels))/n_training
    R = objective - ((eta*(np.sum(np.square(gradient))))/2)
    while(L>R):
        eta = eta/2
        L = (np.sum(np.log(1+np.exp(np.dot(data,(beta-(gradient*eta))))))-np.dot((np.dot(data,(beta-(gradient*eta)))),labels))/n_training
        R = objective - ((eta*(np.sum(np.square(gradient))))/2)
    beta = beta - (eta*gradient)
    objective = (np.sum(np.log(1+np.exp(np.dot(data,beta))))-np.dot((np.dot(data,beta)),labels))/n_training
    cnt +=1
    if(cnt==p):
        p *= 2
        classifier = np.dot(ho_data,beta)
        predicted_labels[np.where(classifier<0)]=0
        predicted_labels[np.where(classifier>=0)]=1
        ho_er = np.sum(ho_labels[:,0]!=predicted_labels)/n_ho
        if((cnt>=32) and(ho_er>(ho_er_best*0.99))):
            break
        ho_er_best = min(ho_er_best,ho_er)
        
    
print('No of iterations executed = {}' .format(cnt))
print('Final Objective Value = {}'.format(objective[0]))
print('Final hold out error rate = {}'.format(ho_er))
