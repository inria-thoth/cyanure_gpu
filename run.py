from cyanure.estimators import Classifier
#from cyanure.data_processing import preprocess
import numpy as np
import scipy.sparse
import time

#load rcv1 dataset about 1Gb, n=781265, p=47152
data = np.load('dataset/covtype.npz')

y=data['arr_1']
X=data['arr_0']

y = np.squeeze(y)


#normalize the rows of X in-place, without performing any copy
#preprocess(X,normalize=True,columns=False)
#declare a binary classifier for squared hinge loss + l1 regularization
initial_time = time.time()
classifier=Classifier(loss='square',penalty="l2",lambda_1=0.000005,max_iter=500,tol=1e-3, duality_gap_interval=10, verbose=False, fit_intercept=False, solver='ista')
# uses the auto solver by default, performs at most 500 epochs
classifier.fit(X,y)
print(time.time() - initial_time)