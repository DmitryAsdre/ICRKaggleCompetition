import os
import pickle
import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from torch.nn import CrossEntropyLoss

from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from hyperopt import fmin, hp, tpe, Trials, space_eval


class CFG:
    DATA_PATH = '../data'
    PATH_TO_SAVE = '../models/'
    
    letter = 'Beta'
    
    exp_name = 'CatboostBaggingROSAlpha_best_params'
    model_name = f'Catboost_{letter}'
    
    hyperopt_space = {'learning_rate' : hp.uniform('learning_rate', 1e-3, 5e-2),
                    'n_estimators' : hp.randint('n_estimators', 200, 1700),
                    'l2_leaf_reg' : hp.choice('l2_leaf_reg', [0.1, 2, 3, 5]),
                    'depth' : hp.randint('depth', 3, 7),
                    'random_strength' : hp.choice('random_strength', [0.5, 1.0, 3.0, 10.0, 15.0])}
    algo = tpe.suggest
    n_step_search = 43
    
    additional_params_hyperopt = {'task_type' : 'GPU', 'devices':'0'}
    additional_params_train = {'task_type' : "CPU"}
    
    n_bag = 100
    random_state = 42
    
def balanced_ce(y_true, y_pred):
    weights = []
    unique = np.sort(list(set(y_true)))
    for i, t in enumerate(unique):
        n_samples_i = np.sum(y_true == t)
        weights.append(1 / (n_samples_i))
    
    
    ce = CrossEntropyLoss()
    y_pred = torch.Tensor(y_pred.astype(np.float32))
    y_true = torch.Tensor(y_true.astype(np.int64)).long()
    
    return ce(y_pred, y_true).item()


class CatboostBaggingROSGreeks:
    def __init__(self, catboost_params, model_name, additional_params={'task_type' : 'GPU', 'devices' : '0'}, n_bag=100, random_state=42):
        self.catboost_params = catboost_params
        self.n_bag = n_bag
        
        self.clf = CatBoostClassifier(**catboost_params, random_state=random_state, verbose=0, **additional_params)
        self.clfs = []
        self.oob_idxs = []
        self.losses = []
        
        self.ros = RandomOverSampler(random_state=random_state)
        self.oob_preds = None
        self.oob_n = None
        self.n_classes = 0
        self.y_pred = None
        self.model_name = model_name
        
    def fit(self, X_train, y_greek):
        self.n_classes = len(set(y_greek))
        
        self.oob_preds = np.zeros((y_greek.shape[0], self.n_classes), dtype=np.float32)
        self.oob_n = np.zeros(y_greek.shape[0], dtype=np.float32) + 1e-20
        
        for i in range(self.n_bag):
            bagged_idxs = []
            while len(set(y_greek.iloc[bagged_idxs])) != self.n_classes:
                bagged_idxs = np.random.randint(0, y_greek.shape[0], y_greek.shape[0])
            oob_idxs = set(list(range(y_greek.shape[0]))) - set(bagged_idxs)
            oob_idxs = list(oob_idxs)
            
            self.oob_idxs.append(oob_idxs)
            
            X_bagged = X_train.iloc[bagged_idxs]
            y_greek_bagged = y_greek.iloc[bagged_idxs]
            
            X_bagged, y_greek_bagged = self.ros.fit_resample(X_bagged, y_greek_bagged)
            
            X_oob = X_train.iloc[oob_idxs]
            
            clf = copy.deepcopy(self.clf)
            clf.fit(X_bagged, y_greek_bagged)
            self.clfs.append(clf)
            
            self.oob_preds[oob_idxs] += clf.predict_proba(X_oob)
            self.oob_n[oob_idxs] += 1
            
            cur_loss = balanced_ce(y_greek, self.oob_preds / self.oob_n.reshape(-1, 1))    
            self.losses.append(cur_loss)
            
        self.y_pred = pd.DataFrame(data=self.oob_preds / self.oob_n.reshape(-1, 1), index=X_train.index, 
                                   columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            
    def predict_oob(self, X_test, df=False):
        oob_preds = np.zeros((X_test.shape[0], self.n_classes), dtype=np.float32)
        oob_n = np.zeros(X_test.shape[0], dtype=np.float32) + 1e-20
        
        for clf, oob_idx in zip(self.clfs, self.oob_idxs):
            X_test_oob = X_test.iloc[oob_idx]
            oob_preds[oob_idx] += clf.predict_proba(X_test_oob)
            oob_n[oob_idx] += 1
        
        if df:
            y_pred = pd.DataFrame(data=oob_preds / oob_n.reshape(-1, 1), index=X_test.index,
                                  columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            return y_pred
        
        return oob_preds / oob_n.reshape(-1, 1)
    
    def predict(self, X_test, df=False):
        preds = np.zeros((X_test.shape[0], self.n_classes), dtype=np.float32)
        
        for clf in self.clfs:
            preds += clf.predict_proba(X_test)
            
        if df:
            y_pred = pd.DataFrame(data=preds / len(self.clfs), index=X_test.index,
                                columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            return y_pred
        
        return preds / len(self.clfs)
    
    
def objective(params):   
    clf = CatboostBaggingROSGreeks(params, model_name=CFG.model_name, n_bag=CFG.n_bag, 
                             random_state=CFG.random_state, additional_params=CFG.additional_params_hyperopt)
    clf.fit(X_train, y_greeks)
    losses = clf.losses
    
    return np.min(losses)
    

if __name__ == '__main__':
    np.random.seed(CFG.random_state)
    
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col='Id')
    greeks = pd.read_csv(os.path.join(CFG.DATA_PATH, 'greeks.csv'), index_col='Id')

    train = train.join(greeks[[CFG.letter]])

    le = LabelEncoder()
    train[CFG.letter] = le.fit_transform(train[CFG.letter])
    
    first_category = train.EJ.unique()[0]
    train.EJ = train.EJ.eq(first_category).astype('int')

    X_train, y_greeks = train.drop(['Class', CFG.letter], axis=1), train[CFG.letter]
    
    best_params = fmin(
                fn=objective,
                space=CFG.hyperopt_space,
                algo=CFG.algo,
                max_evals=CFG.n_step_search)    

    hyperparams = space_eval(CFG.hyperopt_space, best_params)
    print(hyperparams)
    
    clf = CatboostBaggingROSGreeks(hyperparams, n_bag=CFG.n_bag, model_name=CFG.model_name,
                             random_state=CFG.random_state, additional_params=CFG.additional_params_train)
    clf.fit(X_train, y_greeks)

    print(clf.losses[-1])
    
    with open(os.path.join(CFG.PATH_TO_SAVE, CFG.exp_name, f"{CFG.model_name}.pickle"), 'wb') as w:
        pickle.dump(clf, w)
    