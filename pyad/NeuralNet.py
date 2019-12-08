import forward_mode as fwd
import numpy as np

class NeuralNet:
    def __init__(self,n_features,nphl,numhl):
        #n_features = number of features
        #nphl = number of nodes per hidden layer
        #numhl = number of hidden layers
        self.n_features=n_features
        self.nphl=nphl
        self.numhl=numhl
        self.params={}
        
    def init_params(self):
        #weights initialize to random number between 0 and 1
        
        #Parameters for weights between input and first hidden layer and 
        #biases of first hidden layer
        for k in range(self.nphl):
            bname="b_l1_"+str(k+1)
            self.params[bname]=fwd.var(bname,np.random.rand(1)[0])
            for i in range(self.n_features):
                wname="w_l1_"+str(k+1)+str(i+1)
                self.params[wname]=fwd.var(wname,np.random.rand(1)[0])  
        
        #Parameters for weights between hidden layers and 
        #biases of hidden layer (exluding 1st hidden layer which is computed above)
        if self.numhl>1:
            for a in range(self.numhl-1):
                for k in range(self.nphl):
                    bname="b_l"+str(a+2)+"_"+str(k+1)
                    self.params[bname]=fwd.var(bname,np.random.rand(1)[0])
                    for i in range(self.nphl):
                        wname="w_l"+str(a+2)+"_"+str(k+1)+str(i+1)
                        self.params[wname]=fwd.var(wname,np.random.rand(1)[0])  
        
        #Parameters for weights between final hidden layers and 
        #output node and bias of output node                       
        for k in range(self.nphl):
            name="w_out"+str(k+1)
            self.params[name]=fwd.var(name,np.random.rand(1)[0])       
        self.params["b_out"] = fwd.var("b_out",np.random.rand(1)[0])     
        return self.params
        
    def get_output(self,X):
        #dictionary of dictionaries to hold nodes of each hidden layer 
        h_dict={}
        for l in range(self.numhl):
            h_dict[l+1]={}
            
        #Compute nodes of first hidden layer with input from feature vec
        for k in range(self.nphl):
            name = "h_"+str(k+1)
            h_dict[1][name]=fwd.var(name,0)
            for i in range(self.n_features):
                h_dict[1][name]+=self.params["w_l1_"+str(k+1)+str(i+1)]*X[i]   
            h_dict[1][name]+=self.params["b_l1_"+str(k+1)]
            h_dict[1][name]=fwd.logistic(h_dict[1][name])
            
        #Compute nodes of remaining hidden layers
        if self.numhl>1:
            for a in range(self.numhl-1):  
                for k in range(self.nphl):
                    name = "h_"+str(k+1)
                    h_dict[a+2][name]=fwd.var(name,0)
                    for i in range(self.nphl):
                        h_dict[a+2][name]+=self.params["w_l"+str(a+2)+"_"+str(k+1)+str(i+1)]*h_dict[a+1]["h_"+str(i+1)]
                    h_dict[a+2][name]+=self.params["b_l"+str(a+2)+"_"+str(k+1)]
                    h_dict[a+2][name]=fwd.logistic(h_dict[a+2][name])
        
        #compute output node        
        out=fwd.var("out",0)
        for k in range(self.nphl):
                name = "h_"+str(k+1)
                out+=h_dict[self.numhl][name]*self.params["w_out"+str(k+1)]
        out+=self.params["b_out"]
        #use linear activation for final node for regression
        #out=fwd.logistic(out)               
        return out
    
    def forward_pass(self,X,y):
        out=self.get_output(X)
        loss = fwd.abs(y-out)**2
        return loss
    
    def update_weights(self,loss,alpha=.1):
        #Uses gradient descent
        #alpha is the learning rate
        new_params={}
        for k in self.params.keys():
           new_params[k]=self.params[k]-alpha*loss.d[k]
        self.params=new_params   
    
    def train(self,n_epochs,X,y,v=False):  
        for e in range(n_epochs):
            loss_list=[]
            n_obs=len(y)
            for i in range(n_obs):
                X_obs=X[i,]
                y_obs=y[i]
                loss=self.forward_pass(X_obs,y_obs)
                loss_list.append(loss.value)
                self.update_weights(loss) 
            if v==True:     
                print("Mean Loss, epoch {}: {}".format(e+1,sum(loss_list)/len(loss_list)))  

    def score(self,X,y):  
        loss_list=[]
        n_obs=len(y)
        for i in range(n_obs):
            X_obs=X[i,]
            y_obs=y[i]
            loss=self.forward_pass(X_obs,y_obs)
            loss_list.append(loss.value) 
        return(sum(loss_list)/len(loss_list))

    def predict(self,X):  
        pred_list=[]
        n_obs=np.shape(X)[0]
        for i in range(n_obs):
            X_obs=X[i,]
            output=self.get_output(X_obs)
            pred_list.append(output.value) 
        return(pred_list)     
               
#Test on some synthetic data generated from model y=2*x1+3*x2+noise
np.random.seed(123)
X=np.random.rand(50,2)
y_raw=X[:,0]*2+X[:,1]*3+.1*np.random.rand(50,)
y=y_raw/max(y_raw)

X_test=np.random.rand(50,2)
y_raw_test=X_test[:,0]*2+X_test[:,1]*3+.1*np.random.rand(50,)
y_test=y_raw_test/max(y_raw)

testnn=NeuralNet(2,3,2)
testnn.init_params()
  
print("Pre-train loss on train data: ",testnn.score(X,y))
print("Pre-train loss on test data: ",testnn.score(X_test,y_test))   

#Train for 50 epochs
testnn.train(50,X,y,v=True)


print("post-training loss on train data: ",testnn.score(X,y))
print("post-training loss on test data: ",testnn.score(X_test,y_test))   

predictions=testnn.predict(X_test)
for i in range(X_test.shape[0]):
    print("pred: ",predictions[i]," actual: ",y_test[i])


