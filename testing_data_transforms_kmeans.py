import os 

import numpy as np
import pandas as pd
from tqdm import tqdm

from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class CFG:
    DATA_PATH = '../data/'
    n_bag = 100
    
    letters = ['DU', 'AB', 'GL', 'FL', 'CR', 'BQ', 'DA', 'AF', 'BC']
    #letters = ['DU', 'DA', 'CR', 'AB', 'BQ', 'DI', 'DH', 'FI', 'BC']
    n_classes = [3, 4, 5, 8]
    n_pcas = [2, 3, 4, 5]
    tsne_n_components = [1, 2, 3]
    
    cb_params = {'n_estimators' : 1000,
                 'learning_rate' : 5e-2,
                 'l2_leaf_reg' : 0.5,
                 'random_strength' : 3,
                 'auto_class_weights': 'Balanced'}
    

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
    


def try_transformation(X_tr, y_train, cat_features):
    oob_pred = np.zeros(X_tr.shape[0], dtype=np.float32)
    oob_n = np.zeros(X_tr.shape[0], dtype=np.float32) + 1e-20
    
    for i in tqdm(range(CFG.n_bag)):
        bagging_idxs = np.random.randint(0, y_train.shape[0], y_train.shape[0])
        oob_idxs = list(set(list(range(y_train.shape[0]))) - set(bagging_idxs))
        
        y_bagging = y_train.iloc[bagging_idxs]
        X_bagging = X_tr.iloc[bagging_idxs]
        
        clf = CatBoostClassifier(**CFG.cb_params, verbose=0)
        clf.fit(X_bagging, y_bagging, cat_features=cat_features)
        
        X_oob = X_tr.iloc[oob_idxs]
        
        oob_pred[oob_idxs] += clf.predict_proba(X_oob)[:, 1]
        oob_n[oob_idxs] += 1
    
    return balanced_log_loss(y_train, oob_pred / oob_n)

def create_kmeans(X_train, n_clusters, letters_columns):
    X_train = X_train.copy()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    #letters_columns = list(filter(lambda x: x.startswith(letter), X_train.columns))
    X_train_cur = X_train[letters_columns]
    X_train_cur = imp.fit_transform(X_train_cur)
    X_train_cur = std_sc.fit_transform(X_train_cur)
    
    kmeans = KMeans(n_clusters=n_clusters)
    X_transformed = kmeans.fit_transform(X_train_cur)
    
    X_transformed = pd.DataFrame(data=X_transformed, index=X_train.index, 
                                    columns=[f'kmeans_{n_clusters}_{i}' for i in range(n_clusters)])
    #X_train = X_train.join(X_transformed)

    return X_transformed

def create_pca(X_train, n_pca, letters_columns):
    X_train = X_train.copy()
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    
    X_train_cur = X_train[letters_columns]
    
    X_train_cur = imp.fit_transform(X_train_cur)
    X_train_cur = std_sc.fit_transform(X_train_cur)
    
    pca = PCA(n_components=n_pca)
    
    X_transformed = pca.fit_transform(X_train_cur)
    
    X_transformed = pd.DataFrame(data=X_transformed, index=X_train.index, 
                                    columns=[f'pca_{n_pca}_{i}' for i in range(n_pca)])
    #X_train = X_train.join(X_transformed)

    return X_transformed


def create_tsne(X_train, n_components, letters_columns):
    X_train = X_train.copy()
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_sc = StandardScaler()
    
    X_train_cur = X_train[letters_columns]
    
    X_train_cur = imp.fit_transform(X_train_cur)
    X_train_cur = std_sc.fit_transform(X_train_cur)
    
    tsne = TSNE(n_components=n_components)
    
    X_transformed = tsne.fit_transform(X_train_cur)
    
    X_transformed = pd.DataFrame(data=X_transformed, index=X_train.index, 
                                    columns=[f'tsne_{n_components}_{i}' for i in range(n_components)])
    #X_train = X_train.join(X_transformed)

    return X_transformed
    

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(CFG.DATA_PATH, 'train.csv'), index_col=0)
    
    X_train, y_train = train.drop('Class', axis=1), train['Class']
    
    for n_components in CFG.tsne_n_components:
        X_train_tsne = create_tsne(X_train, n_components=n_components, letters_columns=CFG.letters)
        X_train_kmeans = create_kmeans(X_train, n_clusters=5, letters_columns=CFG.letters)

        X_train_ = X_train.join(X_train_tsne).join(X_train_kmeans)
        
        loss = try_transformation(X_train_, y_train, cat_features=['EJ'])

        print(f'Loss : {loss},  letters - {CFG.letters}, n_classes - {5}, tsne_n_components - {n_components}')
    