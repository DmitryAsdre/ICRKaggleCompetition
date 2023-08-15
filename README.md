# Kaggle Competition: ICR - Identifying Age-Related Conditions
This repository contains code for ICR - Identifying Age-Related Conditions Kaggle Competition.

Kaggle Competition - https://www.kaggle.com/competitions/icr-identify-age-related-conditions

<b>My results</b>

I recieved silver medal.

![Alt text](./results/results_icr.png)

## Model
```
Bagged Catboost -> OOB Predict Alpha -> Concat X_train |
Bagged Catboost -> OOB Predict Beta  -> Concat X_train | Meta Bagged ->  Predict
Bagged Catboost -> OOB Predict Gamma -> Concat X_train | Catboost    -> 
Bagged Catboost -> OOB Predict Delta -> Concat X_train |
```