import os 
import copy 
from tqdm import tqdm 

import pandas as pd
import numpy as np

import cuml
from cuml.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier



class CFG:
    DATA_PATH = '../data'
    
    svm_params = {'C' : 100, 'kernel' : 'rbf'}
    
    random_state = 42
    n_bag = 100
    
    
def balanced_log_loss(y_true, y_pred):
    # y_true: correct labels 0, 1
    # y_pred: predicted probabilities of class=1
    # calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # calculate the weights for each class to balance classes
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    # calculate the predicted probabilities for each class
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    # calculate the summed log loss for each class
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    # calculate the weighted summed logarithmic loss
    # (factgor of 2 included to give same result as LL with balanced input)
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    # return the average log loss
    return balanced_log_loss/(N_0+N_1)
    
    
class SVMBagging:
    def __init__(self, svm_params, n_bag=100, random_state=42):
        self.svm_params = svm_params
        self.random_state = random_state
        self.n_bag = n_bag
        self.losses = []
        
    
    def fit(self, X_train, y_train):
        oob_pred = np.zeros(X_train.shape[0], dtype=np.float32)
        oob_n = np.zeros_like(oob_pred) + 1e-20
        
        for i in tqdm(range(self.n_bag)):
            bagging_idxs = np.random.randint(0, X_train.shape[0], X_train.shape[0])
            oob_idxs = set(list(range(X_train.shape[0]))) - set(bagging_idxs)
            oob_idxs = list(oob_idxs)
            
            X_train_bagged = X_train.iloc[bagging_idxs]
            y_train_bagged = y_train.iloc[bagging_idxs]
            
            X_train_oob = X_train.iloc[oob_idxs]
            
            clf = SVC(**self.svm_params, probability=True)
            clf.fit(X_train_bagged, y_train_bagged)
            
            oob_pred[oob_idxs] += clf.predict_proba(X_train_oob)[1]
            oob_n[oob_idxs] += 1
            self.losses.append(balanced_log_loss(y_train, oob_pred/oob_n))
            
    
if __name__ == '__main__':
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col=0)
    simp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    le = LabelEncoder()
    
    train.EJ = le.fit_transform(train.EJ)
    train = pd.DataFrame(data=simp.fit_transform(train), index=train.index, columns=train.columns)
    train = pd.DataFrame(data=std_sc.fit_transform(train), index=train.index, columns=train.columns)
    
    X_train, y_train = train.drop('Class', axis=1).astype(np.float32), train.Class.astype(int)
    
    clf = SVMBagging(CFG.svm_params, CFG.n_bag, CFG.random_state)
    clf.fit(X_train, y_train)
    
    print(clf.losses[-1], np.min(clf.losses))
    
    
            
            