import os
import copy
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from hyperopt import fmin, hp, tpe, Trials, space_eval

import torch
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter
from tg_bot_ml.table_bot import TGTableSummaryWriter

from utils.metrics import balanced_log_loss
from utils.catboost_bagging_ros import CatboostBaggingROS



class CFG:
    device = 'cuda:1'
    DATA_PATH = '../data/'
    
    credentials_path = './credentials.yaml'
    
    binary_target = True
        
    n_step_search = 32
    random_state = 42
    n_bag = 50
    target_id = 'Class'
    
    algo = tpe.suggest
    target_type = 'Alpha'
    MODEL_PATH = '../models/CatboostBaggingROSAlpha_best_params'
    
    log_to_tg = True
    greeks_models = ['Catboost_Alpha.pickle',
                     'Catboost_Beta.pickle',
                     'Catboost_Delta.pickle',
                     'Catboost_Gamma.pickle',]
                     #'Catbost_bagging_tsne_kmeans_Alpha.pickle',
                     #'Catbost_bagging_tsne_kmeans_Beta.pickle',
                     #'Catbost_bagging_tsne_kmeans_Gamma.pickle',
                     #'Catbost_bagging_tsne_kmeans_Delta.pickle']
                     
    dfs = ['../models/TabPFN/TABFPN_bagging_Alpha/prediction.parquet',
           '../models/TabPFN/TABFPN_bagging_Beta/prediction.parquet',
           '../models/TabPFN/TABFPN_bagging_Delta/prediction.parquet',
           '../models/TabPFN/TABFPN_bagging_Gamma/prediction.parquet']
    
    n_kmeans = 5
    n_tsne = 1
    
    letters = ['DU', 'AB', 'GL', 'FL', 'CR', 'BQ', 'DA', 'AF', 'BC']
    
    hyperopt_space = {'learning_rate' : hp.uniform('learning_rate', 1e-3, 5e-2),
                      'n_estimators' : hp.randint('n_estimators', 200, 1800),
                      'l2_leaf_reg' : hp.choice('l2_leaf_reg', [0.1, 2, 3, 5]),
                      'depth' : hp.randint('depth', 3, 7),
                      'random_strength' : hp.choice('random_strength', [0.5, 1.0, 3.0, 10.0])}
    
    exp_name = f'Catboost Bagging - {target_type}'
    additional_params_train = {'task_type' : "CPU"}   
    
    
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
            if CFG.binary_target:
                y_alpha_bagged = y_train.iloc[bagged_idxs]
            else:
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

def objective(params):
    X_train, y_train, y_alpha = train.drop(['Class', 'Alpha'], axis=1), train.Class, train.Alpha
    
    clf = CatboostBaggingROS(params, n_bag=CFG.n_bag, random_state=CFG.random_state)
    clf.fit(X_train, y_train, y_alpha)
    losses = clf.losses
            
    if CFG.log_to_tg:
        tg_writer.add_record(**params, loss=np.min(losses), agmin=np.argmin(losses))
        tg_writer.send(sort_by='loss', ascending=True)
    
    return np.min(losses)


if __name__ == '__main__':
    np.random.seed(CFG.random_state)
    
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col='Id')
    
    #train_kmeans = create_kmeans(train, CFG.n_kmeans, CFG.letters)
    #train_tsne = create_tsne(train, CFG.n_tsne, CFG.letters)
    
    #train = train.join(train_kmeans).join(train_tsne)
    
    for model_name in CFG.greeks_models:
        with open(os.path.join(CFG.MODEL_PATH, model_name), 'rb') as r:
            clf = pickle.load(r)
        y_pred_cur = clf.y_pred
        model_name_ = model_name.split('.')[0]
        y_pred_cur.columns = [f'{model_name_}_{i}' for i in range(y_pred_cur.shape[1])]
        
        train = train.join(y_pred_cur)
    
    for df_name in CFG.dfs:
        df = pd.read_parquet(df_name)
        train = train.join(df)
    
    
    greeks = pd.read_csv(os.path.join(CFG.DATA_PATH, 'greeks.csv'), index_col='Id')
    
    train = train.join(greeks[['Alpha']])
    
    
    first_category = train.EJ.unique()[0]
    train.EJ = train.EJ.eq(first_category).astype('int')
    
    if CFG.log_to_tg:
        tg_writer = TGTableSummaryWriter('./credentials.yaml', CFG.exp_name)
    
    best_params = fmin(
                    fn=objective,
                    space=CFG.hyperopt_space,
                    algo=CFG.algo,
                    max_evals=CFG.n_step_search)    
    
    hyperparams = space_eval(CFG.hyperopt_space, best_params)
    print(hyperparams)
    
    clf = CatboostBaggingROS(hyperparams, CFG.additional_params_train, CFG.n_bag, CFG.random_state)
    X_train, y_train, y_alpha = train.drop(['Class', 'Alpha'], axis=1), train.Class, train.Alpha
    clf.fit(X_train, y_train, y_alpha)
    
    print(clf.losses[-1])
    
    with open(os.path.join(CFG.MODEL_PATH, 'cb_meta_model_binary_tsne_kmeans.pickle'), 'wb') as w:
        pickle.dump(clf, w)
    
