import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn
import utils

def plotFeatures(D, L, features_list, labels_list, mode):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if (i == j):
                plt.figure()
                plt.hist(D[i, L == 0], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.hist(D[i, L == 1], density=True, alpha=0.6, edgecolor='black', linewidth=0.5)
                plt.legend(labels_list)
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+".png")
                plt.close()
            else:
                # Else use scatter plot
                plt.figure()
                plt.scatter(D[i, L == 0], D[j, L == 0], s=5)
                plt.scatter(D[i, L == 1], D[j, L == 1], s=5)
                plt.legend(labels_list)
                plt.ylabel(features_list[j])
                plt.xlabel(features_list[i])
                plt.savefig("./features_analysis/"+mode+"/"+features_list[i]+"-"+features_list[j]+".png")
                plt.close()
    return            

def heatmap(D, L, note=False):
    
    if(note == False):
        folder = "no_an"
    else:
        folder = "an"
    
    # Training set
    plt.figure()
    plt.title("Heatmap of the whole training set", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D)), cmap="Greys", annot=note)
    plt.savefig("./heatmaps/"+folder+"/all.png", dpi=1200)
    plt.close()
    
    # False class
    plt.figure()
    plt.title("Heatmap of the false class", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D[:, L==0])), cmap="Reds", annot=note)
    plt.savefig("./heatmaps/"+folder+"/false.png", dpi=1200)
    plt.close()
    
    # True class
    plt.figure()
    plt.title("Heatmap of the true class", pad = 12.0)
    seaborn.heatmap(abs(np.corrcoef(D[:, L==1])), cmap="Blues", annot=note)
    plt.savefig("./heatmaps/"+folder+"/true.png", dpi=1200)
    plt.close()

    return

def plotDCF(x, y, xlabel, folder_name, text = "", base = 10, ticks = None):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base = base)
    if(ticks is not None):
        plt.xticks(ticks)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    
    plt.ylabel("min DCF")
    plt.title(text, pad = 12.0)
    plt.show()
    # plt.savefig("./"+title+".png", dpi=1200)
    #plt.savefig(folder_name, dpi=1200)
    return



