import pandas as pd
from typing import List, Literal
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from xgboost import XGBRegressor


@dataclass
class ModelConfig:
    """Configuration of XGBoost Regression pipeline"""

    csv_path: Path
    index_col: str
    feature_reduction: bool
    target_feature_y: str
    test_size: float
    seed: int
    columns_to_drop: List = field(default_factory=list)
    trials_param_eval: int = 100
    trials_n_estimators: int = 10000
    trials_loss_metric: Literal["rsme", "r2", "mae"] = "mae"
    recursive_trials: bool = True

    def __read_in_csv__(self):
        self.data = pd.read_csv(self.csv_path, index_col=self.index_col)
        if self.index_col not in self.data.columns:
            raise IndexError(f"Ensure index_col: {self.index_col} present in csv given")


class DataCleaner:
    """Cleans .csv file in preparation for use in XGBoost model"""

    def __init__(self, config: ModelConfig):
        self.config = config

    def type_check_and_replace(self):
        """Convert float columns to int64 if all values are whole numbers."""
        for col in self.config.data.columns:
            if self.config.data[col].dtype in ["float64", "float32"]:
                cells_not_na = self.config.data[col].dropna()
                if len(cells_not_na) > 0 and (cells_not_na % 1 == 0).all():
                    self.config.data[col] = self.config.data[col].astype("int64")

    def target_feature_na_drop(self):
        """Drop rows where target feature has missing values."""
        if self.config.data[self.config.target_feature_y].isna().sum() > 0:
            self.config.data = self.config.data.dropna(
                subset=[self.config.target_feature_y]
            )

    def label_encoding(self):
        """Convert categorical columns to numeric using label encoding."""
        categorical_cols = self.config.data.select_dtypes(include=["object"]).columns
        le = LabelEncoder()
        for col in categorical_cols:
            self.config.data[col] = le.fit_transform(self.config.data[col].astype(str))


class XGBoostRegressor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.trials_hyperparam_space = {
            "max_depth": scope.int(hp.quniform("max_depth", 1, 5, 1)),
            "gamma": hp.uniform("gamma", 0, 1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 50),
            "reg_lambda": hp.uniform("reg_lambda", 10, 100),
            "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
            "min_child_weight": hp.uniform("min_child_weight", 0, 5),
            "learning_rate": hp.uniform("learning_rate", 0, 0.3),
            "max_bin": scope.int(hp.quniform("max_bin", 200, 550, 1)),
            "n_estimators": self.config.trials_n_estimators,
            "random_state": self.config.seed,
            "eval_metric": self.config.trials_loss_metric,
            "early_stopping_rounds": 1000,
        }

    def set_x_y(self):
        self.X = self.config.data.drop(
            columns=self.config.data[self.config.target_feature_y]
        )
        self.y = self.config.data[self.config.target_feature_y]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.config.test_size,
            random_state=self.config.seed,
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=self.config.test_size,
            random_state=self.config.seed,
        )

    def run_trials(self):
        def hyperparam_tuning(space):
            model = XGBRegressor(**self.trials_hyperparam_space)

            evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
            model.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)

            # Predictions and scores
            preds = model.predict(self.X_val)
            rmse = mean_squared_error(self.y_val, preds)
            r2 = r2_score(self.y_val, preds)
            mae = np.mean(np.abs(self.y_val - preds))
            print("SCORE:", rmse, mae, r2)

            loss_value = self.config.trials_loss_metric.strip('"')
            return {"loss": loss_value, "status": STATUS_OK, "model": model}

        trials = Trials()
        self.best = fmin(
            fn=hyperparam_tuning,
            space=self.trials_hyperparam_space,
            algo=tpe.suggest,
            max_evals=self.config.trials_param_eval,
            trials=trials,
        )
        self.best_hyperparams = {
            key: int(value) if key in ["max_depth", "max_bin"] else value
            for key, value in self.best.items()
        }
        self.best_hyperparams = self.trials_hyperparam_space | self.best_hyperparams

    def run_model(self):
        def show_model_stats(XGBmodel):
            preds = XGBmodel.predict(self.X_val)
            mse = mean_squared_error(self.y_val, preds)
            r2 = r2_score(self.y_val, preds)
            mae = np.mean(np.abs(self.y_val - preds))

            print(f"Mean Squared Error: {mse:.2f}")
            print(f"R² Score: {r2:.3f}")
            print(f"Mean Absolute Error: {mae:.2f}")

        def cache_feature_importance(XGBmodel):
            importance = XGBmodel.feature_importances_
            feature_names = self.X_train.columns

            feature_importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            ).sort_values("Importance", ascending=False)

            return feature_importance

        XGBmodel = XGBRegressor(**self.best_hyperparams)
        XGBmodel.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )

        show_model_stats(XGBmodel)
        cache_feature_importance(XGBmodel)


## Rubber ducky:
# I'd like to build a generator that recursively goes over trials, finds the features that = 0
# Then remove those features from the table, and then again apply the trials
# Until it reaches a stage where there are either no more features that = 0
# Or it meets an arbitary 3 - 5 features PER sample (i.e. row) of dataset
