import xgboost
import os
import warnings
import sys
import pandas as pd
import numpy as np
from get_data import read_params
import argparse
import joblib
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    n_estimators=config["estimators"]["random"]["params"]["n_estimators"]
    criterion = config["estimators"]["random"]["params"]["criterion"]
    max_depth= config["estimators"]["random"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["random"]["params"]["min_samples_split"]
    max_features= config["estimators"]["random"]["params"]["max_features"]
    target = [config["base"]["target_col"]]


    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,criterion=criterion,min_samples_split=min_samples_split,max_features=max_features)
    lr.fit(train_x, train_y.values.ravel())

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
             "criterion":criterion,
            "n_estimators": n_estimators,
            "min_samples_split":min_samples_split,
            "max_depth":max_depth,
            "max_features":max_features,
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