''' KNN algorithm '''

''' Step 1 '''

# Part A
# importing libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# Part B
def mean_normalize(X1):
    ''' function for feature scaling using mean'''
    # axis=0 means "along every column"
    X2=(X1-X1.mean(axis=0))/(X1.max(axis=0)-X1.min(axis=0)) 
    return X2


# Part C
def accuracy(y_known,y_predict):
    ''' Gives the accuracy '''
    y_known=y_known.reshape(len(y_known),1)
    y_predict=y_predict.reshape(len(y_predict),1)
    correct=np.count_nonzero(y_known==y_predict)
    total=len(y_known)
    return float(correct)/float(total)
    

# Part D
class Classifier:
    
    def KNN(self,k,X_train,y_train,test_pt):
        ''' K nearest neighbours function, returns the class the test point 
            belongs to. '''
        # For every point in training set X and test point T,  calculated
        T=X_train-test_pt    # Xi-Ti
        T=T**2               # (Xi-Ti)^2 
        S=T.sum(axis=1)      # Σ(Xi-Ti)^2 
        # sqrt(Σ(Xi-Ti)^2) has not been calculated to avoid computation cost
        S=S.reshape(len(S),1)
        D=np.append(S,y_train,axis=1)   # Attaching y_train to S
        
        # Finding k closest points to the test point stpring it in k_pt
        k_pt=D[:k,:]
        for i in range(k,len(D)):
            if np.any(D[i,0]<k_pt[:,0]):
                max_index=np.argmax(k_pt[:,0])
                k_pt[max_index,0]=D[i,0]
                k_pt[max_index,1]=D[i,1]
                
        # Finding that class which majority of these k points belong to
        m=stats.mode(k_pt[:,1])     
        return m.mode[0]
    
    
    def predict(self,k,X_train,y_train,X_test):
        ''' Function for predicting class for X_test '''
        y=[]
        y_train=y_train.reshape(len(y_train),1)
        # For each point in X_test, determining its class and storing it in y
        for i in range(len(X_test)):
            c=self.KNN(k,X_train,y_train,X_test[i,:])
            y.append(c)
        y=np.array(y)
        y=y.reshape(len(y),1)
        return y
    
    
''' Step 2 '''    
# importing datasets
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,4].values


''' Step 3 '''
# feature scaling training and testing sets
X=mean_normalize(X)


''' Step 4 '''
# array for storing accuracy results
acc_result=[]


# cross validation using 4 folds
for i in range(1,5):
    
    # splitting into training and testing sets
    r=range(100*(i-1),100*i)
    X_test=X[r,:]
    y_test=y[r]
    X_train=np.delete(X,r,axis=0)
    y_train=np.delete(y,r,axis=0)
    
    # training and predicting
    knn_classifier=Classifier()
    y_pred=knn_classifier.predict(5,X_train,y_train,X_test)
    
    # getting accuracy
    acc_result.append((accuracy(y_test,y_pred))*100)
    
    
''' Step 5 '''    
# plotting accuracy results
y_pos=np.arange(len(acc_result))
objects=['1','2','3','4']
plt.bar(y_pos,acc_result)
plt.xticks(y_pos,objects)
plt.title('Accuracy results')
plt.xlabel('Round')
plt.ylim(75,100)
plt.ylabel('Accuracy (%)')
plt.show()
    

    





        
        
        
                
        
                
                
                
        
    
    