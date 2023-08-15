import os
import pickle
import copy
import sys
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import torch
from torch.nn import CrossEntropyLoss

from tabpfn import TabPFNClassifier
from imblearn.over_sampling import RandomOverSampler
from hyperopt import fmin, hp, tpe, Trials, space_eval


class CFG:
    DATA_PATH = '../data'
    PATH_TO_SAVE = '../models/TabPFN'
    
    letter = 'Alpha'
    
    exp_name = 'CatboostBaggingROSAlpha_best_params'
    model_name = f'TABFPN_bagging'
    
    device = 'cuda:1'
    N_ensemble_configurations = 12
    
    n_bag = 30
    random_state = 42
    
    #n_tsne = 1
    #n_kmeans = 5
    #letters = ['DU', 'AB', 'GL', 'FL', 'CR', 'BQ', 'DA', 'AF', 'BC']
    
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


class TabfpnBaggingROSGreeks:
    def __init__(self, model_name, PATH_TO_SAVE, n_bag=100, device='cuda:0', N_ensemble_configurations=24, random_state=42):
        self.PATH_TO_SAVE = PATH_TO_SAVE
        self.n_bag = n_bag
        self.oob_idxs = []
        self.losses = []
        
        os.makedirs(os.path.join(CFG.PATH_TO_SAVE, model_name), exist_ok=True)
        
        self.ros = RandomOverSampler(random_state=random_state)
        self.oob_preds = None
        self.oob_n = None
        self.n_classes = 0
        self.y_pred = None
        self.model_name = model_name
        self.device = device
        self.N_ensemble_configurations = N_ensemble_configurations
        
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
            
            clf = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.N_ensemble_configurations)
            clf.fit(X_bagged, y_greek_bagged, overwrite_warning=True)
            
            with open(os.path.join(self.PATH_TO_SAVE, self.model_name, f'model_{i}.pickle'), 'wb') as w:
                pickle.dump(clf, w)
            
            self.oob_preds[oob_idxs] += clf.predict_proba(X_oob)
            self.oob_n[oob_idxs] += 1
            
            cur_loss = balanced_ce(y_greek, self.oob_preds / self.oob_n.reshape(-1, 1))    
            self.losses.append(cur_loss)
            
            del clf
            gc.collect()
            
        self.y_pred = pd.DataFrame(data=self.oob_preds / self.oob_n.reshape(-1, 1), index=X_train.index, 
                                   columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            
    def predict_oob(self, X_test, model_path, df=False):
        oob_preds = np.zeros((X_test.shape[0], self.n_classes), dtype=np.float32)
        oob_n = np.zeros(X_test.shape[0], dtype=np.float32) + 1e-20
        
        for i, oob_idx in enumerate(self.oob_idxs):
            X_test_oob = X_test.iloc[oob_idx]
            with open(os.path.join(model_path, f'model_{i}.pickle'), 'rb') as r:
                clf = pickle.load(r)
            oob_preds[oob_idx] += clf.predict_proba(X_test_oob)
            oob_n[oob_idx] += 1
        
        if df:
            y_pred = pd.DataFrame(data=oob_preds / oob_n.reshape(-1, 1), index=X_test.index,
                                  columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            return y_pred
        
        return oob_preds / oob_n.reshape(-1, 1)
    
    def predict(self, X_test, model_path, df=False):
        preds = np.zeros((X_test.shape[0], self.n_classes), dtype=np.float32)
        
        for i in range(self.n_bag):
            with open(os.path.join(model_path, f'model_{i}.pickle'), 'rb') as r:
                clf = pickle.load(r)
            preds += clf.predict_proba(X_test)
            
        if df:
            y_pred = pd.DataFrame(data=preds / len(self.clfs), index=X_test.index,
                                columns=[f"{self.model_name}_{i}" for i in range(self.n_classes)])
            return y_pred
        
        return preds / len(self.clfs)
    
def create_kmeans(X_train, n_clusters, letters_columns):
    X_train = X_train.copy()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    #letters_columns = list(filter(lambda x: x.startswith(letter), X_train.columns))
    X_train_cur = X_train[letters_columns]
    X_train_cur = imp.fit_transform(X_train_cur)
    X_train_cur = std_sc.fit_transform(X_train_cur)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=CFG.random_state)
    X_transformed = kmeans.fit_transform(X_train_cur)
    
    X_transformed = pd.DataFrame(data=X_transformed, index=X_train.index, 
                                    columns=[f'kmeans_{n_clusters}_{i}' for i in range(n_clusters)])
    #X_train = X_train.join(X_transformed)

    return X_transformed


def create_tsne(X_train, n_components, letters_columns):
    X_train = X_train.copy()
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    
    X_train_cur = X_train[letters_columns]
    
    X_train_cur = imp.fit_transform(X_train_cur)
    X_train_cur = std_sc.fit_transform(X_train_cur)
    
    tsne = TSNE(n_components=n_components, random_state=CFG.random_state)
    
    X_transformed = tsne.fit_transform(X_train_cur)
    
    X_transformed = pd.DataFrame(data=X_transformed, index=X_train.index, 
                                    columns=[f'tsne_{n_components}_{i}' for i in range(n_components)])
    #X_train = X_train.join(X_transformed)

    return X_transformed
    
    
def objective(params):   
    #clf = CatboostBaggingROSGreeks(params, model_name=CFG.model_name, n_bag=CFG.n_bag, 
    #                         random_state=CFG.random_state, additional_params=CFG.additional_params_hyperopt)
    #clf.fit(X_train, y_greeks)
    #losses = clf.losses
    
    #return np.min(losses)
    pass
    

if __name__ == '__main__':
    CFG.letter = sys.argv[1]
    
    CFG.model_name = f'{CFG.model_name}_{CFG.letter}'

    np.random.seed(CFG.random_state)
    
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col='Id')
    greeks = pd.read_csv(os.path.join(CFG.DATA_PATH, 'greeks.csv'), index_col='Id')

    train = train.join(greeks[[CFG.letter]])

    le = LabelEncoder()
    train[CFG.letter] = le.fit_transform(train[CFG.letter])
    
    first_category = train.EJ.unique()[0]
    train.EJ = train.EJ.eq(first_category).astype('int')

    X_train, y_greeks = train.drop(['Class', CFG.letter], axis=1), train[CFG.letter]   

    clf = TabfpnBaggingROSGreeks(CFG.model_name, CFG.PATH_TO_SAVE, CFG.n_bag, device=CFG.device, 
                                 N_ensemble_configurations=CFG.N_ensemble_configurations, random_state=CFG.random_state)
    clf.fit(X_train, y_greeks)
    print(clf.losses[-1])
    
    with open(os.path.join(CFG.PATH_TO_SAVE, CFG.model_name, f"meta_learner.pickle"), 'wb') as w:
        pickle.dump(clf, w)
    
    clf.y_pred.to_parquet(os.path.join(CFG.PATH_TO_SAVE, CFG.model_name, 'prediction.parquet'))