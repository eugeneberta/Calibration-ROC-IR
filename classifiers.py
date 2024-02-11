import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

seed = 1

# Parameters for the training of our classifiers:
train_size = 4500
cal_size = 500 # 10% of the data used for calibration
test_size = 1000

# We use a filtered version of the full covertype dataset, we keep only the first 1000 rows for the first 4 cover types:
df = pd.read_csv("./covtypesmall.csv")

# binary classifier:
data = df[df.Class.isin([1,2])].to_numpy()

np.random.seed(seed)
np.random.shuffle(data)

X = np.array(data[:,1:-1], dtype=float)
Y = np.array(data[:,-1])

Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xcal, Ycal = X[train_size:train_size+cal_size], Y[train_size:train_size+cal_size]
Xtest, Ytest = X[train_size+cal_size:train_size+cal_size+test_size], Y[train_size+cal_size:train_size+cal_size+test_size]

lr = LogisticRegression(random_state=seed).fit(Xtrain, Ytrain)
Pcal = lr.predict_proba(Xcal)
Ptest = lr.predict_proba(Xtest)

Ycal_ = np.zeros((cal_size, 2))
Ycal_[:,0] = np.array(Ycal == lr.classes_[0])
Ycal_[:,1] = np.array(Ycal == lr.classes_[1])

Ytest_ = np.zeros((test_size, 2))
Ytest_[:,0] = np.array(Ytest == lr.classes_[0])
Ytest_[:,1] = np.array(Ytest == lr.classes_[1])

np.save('predictions/Cover2Pcal.npy', Pcal)
np.save('predictions/Cover2Ycal.npy', Ycal_)
np.save('predictions/Cover2Ptest.npy', Ptest)
np.save('predictions/Cover2Ytest.npy', Ytest_)

# K=3 classifer:
data = df[df.Class.isin([1,2,3])].to_numpy()

np.random.seed(seed)
np.random.shuffle(data)

X = np.array(data[:,1:-1], dtype=float)
Y = np.array(data[:,-1])

Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xcal, Ycal = X[train_size:train_size+cal_size], Y[train_size:train_size+cal_size]
Xtest, Ytest = X[train_size+cal_size:train_size+cal_size+test_size], Y[train_size+cal_size:train_size+cal_size+test_size]

clf = LogisticRegression(random_state=seed).fit(Xtrain, Ytrain)

Pcal = clf.predict_proba(Xcal)
Ptest = clf.predict_proba(Xtest)

Ycal_ = np.zeros((cal_size, 3))
Ycal_[:,0] = np.array(Ycal == clf.classes_[0])
Ycal_[:,1] = np.array(Ycal == clf.classes_[1])
Ycal_[:,2] = np.array(Ycal == clf.classes_[2])

Ytest_ = np.zeros((test_size, 3))
Ytest_[:,0] = np.array(Ytest == clf.classes_[0])
Ytest_[:,1] = np.array(Ytest == clf.classes_[1])
Ytest_[:,2] = np.array(Ytest == clf.classes_[2])

np.save('predictions/Cover3Pcal.npy', Pcal)
np.save('predictions/Cover3Ycal.npy', Ycal_)
np.save('predictions/Cover3Ptest.npy', Ptest)
np.save('predictions/Cover3Ytest.npy', Ytest_)

# K=4 classifer:
data = df[df.Class.isin([1,2,3,4])].to_numpy()

np.random.seed(seed)
np.random.shuffle(data)

X = np.array(data[:,1:-1], dtype=float)
Y = np.array(data[:,-1])

Xtrain, Ytrain = X[:train_size], Y[:train_size]
Xcal, Ycal = X[train_size:train_size+cal_size], Y[train_size:train_size+cal_size]
Xtest, Ytest = X[train_size+cal_size:train_size+cal_size+test_size], Y[train_size+cal_size:train_size+cal_size+test_size]

clf = LogisticRegression(random_state=seed).fit(Xtrain, Ytrain)

Pcal = clf.predict_proba(Xcal)
Ptest = clf.predict_proba(Xtest)

Ycal_ = np.zeros((cal_size, 4))
Ycal_[:,0] = np.array(Ycal == clf.classes_[0])
Ycal_[:,1] = np.array(Ycal == clf.classes_[1])
Ycal_[:,2] = np.array(Ycal == clf.classes_[2])
Ycal_[:,3] = np.array(Ycal == clf.classes_[3])

Ytest_ = np.zeros((test_size, 4))
Ytest_[:,0] = np.array(Ytest == clf.classes_[0])
Ytest_[:,1] = np.array(Ytest == clf.classes_[1])
Ytest_[:,2] = np.array(Ytest == clf.classes_[2])
Ytest_[:,3] = np.array(Ytest == clf.classes_[3])

np.save('predictions/Cover4Pcal.npy', Pcal)
np.save('predictions/Cover4Ycal.npy', Ycal_)
np.save('predictions/Cover4Ptest.npy', Ptest)
np.save('predictions/Cover4Ytest.npy', Ytest_)