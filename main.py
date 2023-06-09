import utils
import plot
import numpy as np
import gaussianClassifier as gc
import logisticRegression as lr



# Load the training data
D, L = utils.load("data/Train.txt")

print("Number of training samples for the male class: " + str(D[:, L==0].shape[1]))
print("Number of training samples for the female class: " + str(D[:, L==1].shape[1]))
print("Number of features: " + str(D.shape[0]))

# Features exploration
#plot.plotFeatures(D, L, utils.features_list, utils.classes_list, "original")
normalizedData, normalizedMean, normalizedStandardDeviation = utils.ZNormalization(D)
normalizedData = utils.Gaussianization(normalizedData)
#plot.plotFeatures(normalizedData, L, utils.features_list, utils.classes_list, "normalized-gaussianized")

# Correlation analysis
#plot.heatmap(normalizedData, L, True)
#plot.heatmap(normalizedData, L, False)

# MVG CLASSIFIER 
# gc.computeMVGClassifier(normalizedData, L)


# LOGISTIC REGRESSION 
# lr.findBestLambda(normalizedData, L)

lambd_lr = 1e-4 # best value of lambda

lr.computeLogisticRegression(normalizedData, L, lambd = lambd_lr)



