import matplotlib.pyplot as plt
import numpy 
from Library.lib import * 
import seaborn as sb
from scipy.stats import norm
import logistic_Regression as lr



########################## Loading Datasets ##########################################
DTR, LTR = load('data/Train.txt')
DTE, LTE = load('data/Test.txt')



def features_Analysis(DTR,LTR,DTE):
    L0_DTR = DTR[:,LTR==0] #Male class
    L1_DTR = DTR[:,LTR==1] #Female class
    
    print("The number of Training samples for the class Male are "+str(L0_DTR.shape[1]))
    print("The number of Training samples for the class Female are "+str(L1_DTR.shape[1]))
    
    print("The number of Features are "+str(DTR.shape[0]))
    
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_DTR[i,:],bins = 40,density=True,label='Male',edgecolor='b',alpha=0.65)
        plt.hist(L1_DTR[i,:],bins = 40,density=True,label='Female',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Gaussianize the Training features 
    y = numpy.zeros(DTR.shape)
    
    for i in range(DTR.shape[1]) :
        x = DTR[:,i]
        count = numpy.sum(x.reshape((x.shape[0],1))>DTR,axis=1)
        rank = (count+1)/ (DTR.shape[1]+2)
        y[:,i]= norm.ppf(rank)
     
    
    L0_ = y[:,LTR==0] #Male class
    L1_ = y[:,LTR==1] #Female class
    
    
    for i in range(DTR.shape[0]):
        
        plt.figure('features Distribution')
        plt.xlabel("Feature"+str(i+1))
        plt.ylabel("Contibution")
        plt.hist(L0_[i,:],bins = 40,density=True,label='Male',edgecolor='b',alpha=0.65)
        plt.hist(L1_[i,:],bins = 40,density=True,label='Female',edgecolor='r',alpha=0.65)
        plt.legend(loc='upper right')
        plt.show()   
   
    # Displaying heatmap
    allDataset = abs(numpy.corrcoef(DTR))
    sb.heatmap(allDataset, cmap="Greys", annot=True)
    plt.xlabel("Total Data Features Correlation")
    plt.ylabel("Raw Data")
    
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_DTR))
    sb.heatmap(posSamples, cmap="Reds", annot=True)
    plt.xlabel("Female Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_DTR))
    sb.heatmap(negSamples, cmap="Blues", annot=True)
    plt.xlabel("Male Class Freatures Correlation")
    plt.ylabel("Raw Data")
    plt.show()
    
    # displaying heatmap after Gaussianization
   
    allDataset = abs(numpy.corrcoef(y))
    sb.heatmap(allDataset, cmap="Greys", annot=True)
    plt.xlabel("Total Data Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    posSamples = abs(numpy.corrcoef(L1_))
    sb.heatmap(posSamples, cmap="Reds", annot=True)
    plt.xlabel("Female Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    negSamples = abs(numpy.corrcoef(L0_))
    sb.heatmap(negSamples, cmap="Blues", annot=True)
    plt.xlabel("Male Class Freatures Correlation")
    plt.ylabel("Gaussianized Data")
    plt.show()
    
    DTR=y
    return DTR,DTE

         

#------------------- main----------------------------#

# Feature analysis and Gaussianizing the training 
DTR_,DTE_ = features_Analysis(DTR,LTR,DTE)

DTR,p,m = PCA_optimal(DTR_,0.95) 

# (DTR, LTR), (DTE, LTE) = split_db_2to1(DTR, LTR)
priors = [0.1,0.5,0.9]
print("\n######### Logistic Regression ############")
start = 1e-5
stop = 1e5
num_points = 100

step = (stop - start) / (num_points - 1)
lmd = numpy.logspace(numpy.log10(start), numpy.log10(stop),num_points)
tot_minDCF = numpy.zeros((len(priors),len(lmd)),dtype= numpy.float32 )
k=0
for i in priors:   
    print("\n\nApplication with prior of "+str(i))
    print("------------------------------------")
    print("K-fold")
    print("-------------------------\n")
    
    for j in range(lmd.shape[0]):
        print("Lambda : "+str(lmd[j]))
        print("-------------------------")
        args = [i,lmd[j],5]
        evaluation_acc,t = Kfold(lr, DTR, LTR,args)
        
        tot_minDCF[k,j] = t
    k+=1
plt.plot(lmd, tot_minDCF[0], label='prior of 0.1', color = 'green')
plt.plot(lmd, tot_minDCF[1], label='prior of 0.5', color = 'red')
plt.plot(lmd, tot_minDCF[2], label='prior of 0.9', color = 'blue')
plt.legend(loc="upper left")
plt.xlabel('Lamda')
plt.ylabel('minDCF LR')
plt.xscale('log')
plt.show()