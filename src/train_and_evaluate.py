import xgboost
import os
import warnings
import sys
import pandas as pd
import numpy as np
from get_data import read_params
import argparse
import joblib
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    learning_rate=config["estimators"]["xgboost"]["params"]["learning_rate"]
    n_estimators= config["estimators"]["xgboost"]["params"]["n_estimators"]
    max_depth = config["estimators"]["xgboost"]["params"]["max_depth"]
    min_child_weight = config["estimators"]["xgboost"]["params"]["min_child_weight"]
    gamma = config["estimators"]["xgboost"]["params"]["gamma"]
    subsample = config["estimators"]["xgboost"]["params"]["subsample"]
    colsample_bytree = config["estimators"]["xgboost"]["params"]["colsample_bytree"]
    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = xgboost.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight,gamma=gamma,
      subsample=subsample, colsample_bytree=colsample_bytree,
        random_state=random_state)
    lr.fit(train_x, train_y)

    x_predicted = lr.predict(test_x)
    accuracy_scor=accuracy_score(test_y,x_predicted)
    r2_scor=r2_score(test_y,x_predicted)

    #####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "accuracy_score": accuracy_scor,
            "r2_score": r2_scor,
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth":max_depth,
            "min_child_weight":min_child_weight,
            "gamma":gamma,
             "subsample":subsample,
            "colsample_bytree":colsample_bytree,
            "random_state":random_state
        }
        json.dump(params, f, indent=4)
    #####################################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)