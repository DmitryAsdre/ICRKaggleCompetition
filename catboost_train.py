import os
import pickle
import copy

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler


class CFG:
    DATA_PATH = '../data'
    PATH_TO_SAVE = '../models/'
    
    exp_name = 'CatboostBaggingROSAlpha_best_params'
    
    cb_params = {"depth" : 3,
                 "l2_leaf_reg" : 2,
                 'learning_rate' : 0.0275232,
                 'n_estimators' : 831,
                 'random_strength' : 10,
                 'border_count' : 255}
    additional_params={'task_type' : 'CPU'}
    n_bag = 100
    random_state = 42
    

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

class CatboostBaggingROS:
    def __init__(self, catboost_params, additional_params={'task_type' : 'GPU', 'devices' : '0'}, n_bag=100, random_state=42):
        self.catboost_params = catboost_params
        self.n_bag = n_bag
        
        self.clf = CatBoostClassifier(**catboost_params, random_state=random_state, verbose=0, **additional_params)
        self.clfs = []
        self.oob_idxs = []
        self.losses = []
        
        self.ros = RandomOverSampler(random_state=random_state)
        self.oob_preds = None
        self.oob_n = None
        
    def fit(self, X_train, y_train, y_alpha):
        self.oob_preds = np.zeros(y_train.shape[0], dtype=np.float32)
        self.oob_n = np.zeros(y_train.shape[0], dtype=np.float32) + 1e-20
        
        for i in range(self.n_bag):
            bagged_idxs = np.random.randint(0, y_train.shape[0], y_train.shape[0])
            oob_idxs = set(list(range(y_train.shape[0]))) - set(bagged_idxs)
            oob_idxs = list(oob_idxs)
            
            self.oob_idxs.append(oob_idxs)
            
            X_bagged = X_train.iloc[bagged_idxs]
            y_alpha_bagged = y_alpha.iloc[bagged_idxs]
            
            X_bagged, y_alpha_bagged = self.ros.fit_resample(X_bagged, y_alpha_bagged)
            
            X_oob = X_train.iloc[oob_idxs]
            
            clf = copy.deepcopy(self.clf)
            clf.fit(X_bagged, y_alpha_bagged)
            self.clfs.append(clf)
            
            self.oob_preds[oob_idxs] += clf.predict_proba(X_oob)[:, 1:].sum(axis=1)
            self.oob_n[oob_idxs] += 1
            
            cur_loss = balanced_log_loss(y_train, self.oob_preds / self.oob_n)            
            self.losses.append(cur_loss)
            
        
    def predict(self, X_test, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_bag
        
        y_pred = np.zeros(X_test.shape[0], dtype=np.float32)   
        
        for i in range(n_estimators):
            clf = self.clfs[i]
            y_pred += clf.predict_proba(X_test)[:, 1:].sum(axis=1)
            
        y_pred /= n_estimators
        
        return y_pred
    
    def predict_oob(self, X_test, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_bag
            
        y_pred = np.zeros(X_test.shape[0], dtype=np.float32)
        oob_n = np.zeros_like(y_pred)
        for i in range(n_estimators):
            clf = self.clfs[i]
            oob_idxs = self.oob_idxs[i]
            y_pred[oob_idxs] += clf.predict_proba(X_test.iloc[oob_idxs])[:, 1:].sum(axis=1)
            oob_n[oob_idxs] += 1
            
        y_pred /= oob_n
        
        return y_pred
    

if __name__ == '__main__':
    np.random.seed(CFG.random_state)
    
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col='Id')
    greeks = pd.read_csv(os.path.join(CFG.DATA_PATH, 'greeks.csv'), index_col='Id')
    
    train = train.join(greeks[['Alpha']])
    
    
    first_category = train.EJ.unique()[0]
    train.EJ = train.EJ.eq(first_category).astype('int')
    
    X_train, y_train, y_alpha = train.drop(['Class', 'Alpha'], axis=1), train.Class, train.Alpha

    
    clf = CatboostBaggingROS(CFG.cb_params, additional_params=CFG.additional_params, n_bag=CFG.n_bag)
    
    clf.fit(X_train, y_train, y_alpha)
    
    os.makedirs(os.path.join(CFG.PATH_TO_SAVE, CFG.exp_name), exist_ok=True)
    
    with open(os.path.join(CFG.PATH_TO_SAVE, CFG.exp_name, 'cb_bagging.pickle'), 'wb') as w:
        pickle.dump(clf, w)
    
    y_pred = clf.predict_oob(X_train)
    y_pred_ = clf.predict(X_train)
    
    print(balanced_log_loss(y_train, y_pred))
    print(balanced_log_loss(y_train, y_pred_))
    print(clf.losses[-1])
    