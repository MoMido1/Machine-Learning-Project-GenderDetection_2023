import numpy
import scipy
import scipy.linalg
import math


def load(File):
    Data = []
    Labels = []
    with open(File) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1]
                Data.append(attrs)
                Labels.append(label)
            except:
                pass    
    Data = numpy.hstack(Data)
    Labels = numpy.array(Labels, dtype=numpy.int32)
    return Data, Labels


def mcol (lst):
    return lst.reshape((lst.shape[0],1))

def mrow (lst):
    return lst.reshape((1,lst.shape[0]))
def PCA_optimal (DataMat, t):
    mu = DataMat.mean(1)
    mu = mcol(mu)
    DataCentered = DataMat - mu
    Cov = numpy.dot(DataCentered,DataCentered.T)
    Cov = Cov / DataCentered.shape[1]
    U, s,_ = numpy.linalg.svd(Cov)
    tot_eigVal = numpy.sum(s)
    
    min_m =0
    eigval_cum =0
    
    for i in range(s.shape[0]) :
        eigval_cum += s[i]
        if(eigval_cum/tot_eigVal >= t):
            min_m = i+1
            break  
        
    P = U[: , 0:min_m]
    DataProjected = numpy.dot(P.T,DataMat)

    return DataProjected ,P, min_m



def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)       
    return sorted(unique_list)

def DCF (p1,Cfn,Cfp,p00,p01,p10,p11):
    FNR = p01/(p01+p11)
    FPR = p10/(p00+p10)
    
    return p1*Cfn*FNR+(1-p1)*Cfp*FPR

def Bdummy (p1,Cfn,Cfp):
    par1= p1*Cfn
    par2= (1-p1)*Cfp
    
    return par1 if par1<par2 else par2


def Norm_DCF (DCF,Bmin):
    return DCF/Bmin

def Threshold (p1,Cfn,Cfp):
    t = - math.log((p1*Cfn)/((1-p1)*Cfp))
    return t

def Min_DCF(llr,labels,p1):
    Cfn=1
    Cfp=1
    sortedLLR= sorted(llr)
    T =[min(sortedLLR)-1, *sortedLLR, max(sortedLLR)+1]
    DCFall=2
    u=0
    
    for t in T:
        bayasPrediction_t= llr>t
        p00=0
        p01=0
        p10=0
        p11=0
        
        p00=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t ==0))
        p10=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t !=0))
        p01=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t ==0))
        p11=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t !=0))
        dcf = DCF(p1,Cfn,Cfp,p00,p01,p10,p11)
        
        if dcf<DCFall:
            DCFall= dcf
        u+=1     
    b=Bdummy (p1,Cfn,Cfp)   
    return Norm_DCF(DCFall, b)

def Act_DCF(llr,labels,p1):
    Cfn=1
    Cfp=1
    bayasPrediction_t= llr>Threshold(p1, Cfn, Cfp)
    
    p00=0
    p01=0
    p10=0
    p11=0
    
    p00=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t ==0))
    p10=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t !=0))
    p01=numpy.sum((bayasPrediction_t != labels) & (bayasPrediction_t ==0))
    p11=numpy.sum((bayasPrediction_t == labels) & (bayasPrediction_t !=0))
    
    b=Bdummy (p1,Cfn,Cfp)    
    dcf=DCF(p1,Cfn,Cfp,p00,p01,p10,p11)
    
    return Norm_DCF(dcf, b)

def Compute_Accuracy(Y, Y_predict):
    compare = Y_predict[Y_predict == Y]
    accuracy = compare.shape[0]/Y.shape[0] *100
    return accuracy

def Kfold(model, X, Y, args):
    prior= args[0]
    K = args[2]
    n_attributes, n_samples = X.shape
    X = X.T
    n_samples_per_fold = int(n_samples/K)
    starting_index = 0
    ending_index = n_samples_per_fold
    total_accuracy = 0
    min_Dcf=2
    
    for i in range(K):
        # Compute the testing samples
        X_test = X[starting_index : ending_index]
        Y_test = Y[starting_index : ending_index]
        
        # Compute the training samples
        X_train_part1 = X[0 : starting_index]
        X_train_part2 = X[ending_index: -1]
        X_train = numpy.concatenate((X_train_part1, X_train_part2), axis = 0)
        
        Y_train_part1 = Y[0 : starting_index]
        Y_train_part2 = Y[ending_index: -1]
        Y_train = numpy.concatenate((Y_train_part1, Y_train_part2), axis = 0)
        
        # Apply to the model and get accuracy
        ret_args = model.train(X_train.T, Y_train,args)

        
        results,llr = model.evaluate(X_test.T,ret_args)
        total_accuracy += Compute_Accuracy(Y_test, results)
        # try:
        act_dcf = Min_DCF(llr,Y_test,prior)
        min_Dcf =  act_dcf if act_dcf < min_Dcf else min_Dcf
        # except:
        #     pass
        
        # Updating indexes for next iteration
        starting_index += n_samples_per_fold
        ending_index += n_samples_per_fold
        
    avg_accuracy = total_accuracy/K
    print("Evaluation Average Accuracy = ", round(avg_accuracy, 1)," %")
    
    # avg_Dcf = total_Dcf/K
    print("MinDCF for Kfold Evaluation = "+str(min_Dcf))
    
    return avg_accuracy,min_Dcf
