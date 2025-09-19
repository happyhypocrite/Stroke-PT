from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from alive_progress import alive_bar


@dataclass
class ModelConfig:
    """Configuration of XGBoost Regression pipeline"""

    csv_path: str
    index_col: str
    target_feature_y: str
    columns_to_drop: List = field(default_factory=list)
    seed: int = 42
    test_size: float = 0.2
    trials_param_eval: int = 100
    trials_n_estimators: int = 10000
    trials_loss_metric: Literal["rmse", "mae"] = "mae"
    recursive_trials: bool = True
    min_features_per_sample: int = 3


class DataCleaner:
    """Cleans .csv file in preparation for use in XGBoost model.

    Args:
        config (ModelConfig): Configuration object containing data processing parameters.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def read_in_csv(self):
        self.data = pd.read_csv(self.config.csv_path, index_col=self.config.index_col)
        if self.config.index_col not in pd.read_csv(self.config.csv_path).columns:
            raise IndexError(
                f"Ensure index_col: {self.config.index_col} present in csv given"
            )

    def type_check_and_replace(self):
        """Convert float columns to int64 if all values are whole numbers.

        Returns:
            None: Modifies self.data in place.
        """

        for col in self.data.columns:
            if self.data[col].dtype in ["float64", "float32"]:
                cells_not_na = self.data[col].dropna()
                if len(cells_not_na) > 0 and (cells_not_na % 1 == 0).all():
                    self.data[col] = self.data[col].astype("Int64")

    def drop_user_cols(self):
        cols_to_drop = [
            col for col in self.config.columns_to_drop if col in self.data.columns
        ]
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
        else:
            raise KeyError(
                "Please ensure columns specified to be dropped are present in the dataset."
            )

    def target_feature_na_drop(self):
        """Drop rows where target feature has missing values.

        Returns:
            None: Modifies self.data in place.
        """

        if self.data[self.config.target_feature_y].isna().sum() > 0:
            self.data = self.data.dropna(subset=[self.config.target_feature_y])

    def label_encoding(self):
        """Convert categorical columns to numeric using label encoding.

        Returns:
            None: Modifies self.data in place.
        """

        categorical_cols = self.data.select_dtypes(include=["object"]).columns
        le = LabelEncoder()
        for col in categorical_cols:
            self.data[col] = le.fit_transform(self.data[col].astype(str))


class DataStorage:
    """Collects included features, models stats across feature exclusion iterations
    Collects included feattures, chosen hyperparameters, final model stats, and feature importance
    from chosen hyperparameter/selected feature combination.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.iteration_data = {}

    def store_iteration(
        self,
        iteration_num: int,
        feature_importance_df: pd.DataFrame,
        mae: float,
        r2: float,
        rmse: float,
    ):
        """Store feature importance data for a specific iteration"""
        iteration_df = feature_importance_df.copy()
        iteration_df["MAE"] = mae
        iteration_df["R2"] = r2
        iteration_df["RMSE"] = rmse

        self.iteration_data[iteration_num] = (
            iteration_df  # Key: Iteration number, Value: pd.DataFrame of stats and features
        )

    def store_final_model_stats(
        self,
        feature_importance_df: pd.DataFrame,
        tuned_hyperparams: dict,
        mae: float,
        r2: float,
        rmse: float,
    ):
        final_model_df = feature_importance_df.copy()
        final_model_df["MAE"] = mae
        final_model_df["R2"] = r2
        final_model_df["RMSE"] = rmse
        for param, value in tuned_hyperparams.items():
            final_model_df[param] = value
        self.final_model_data = final_model_df

    def save_final_and_iter_model_stats(self):
        """Save iteration data and final model stats to Excel file with separate sheets.

        Args:
            filename (str): Output Excel filename

        Returns:
            None: Creates Excel file with multiple sheets
        """
        with pd.ExcelWriter("XGBoost_Model_Results.xlsx", engine="openpyxl") as writer:
            # Save each iteration as a separate sheet
            for iteration_num, df in self.iteration_data.items():
                sheet_name = f"Iteration_{iteration_num}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            if hasattr(self, "final_model_data"):
                self.final_model_data.to_excel(
                    writer, sheet_name="Final_Model", index=False
                )


class ModelOptimisation:
    """Handles XGBoost model hyperparameter optimisation.

    Args:
        config (ModelConfig): Configuration object containing model parameters.

    Attributes:
        trials_hyperparam_space (dict): Hyperparameter search space for optimisation.
        tuned_hyperparams (dict): Optimized hyperparameters after tuning.
    """

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

    def set_x_y(self, data):
        """Splits data into features and target, then creates train/validation/test sets.

        Args:
            data (pd.DataFrame): Input dataset containing features and target variable.

        Returns:
            None: Updates instance attributes with train/validation/test splits.
        """

        self.X = data.drop(columns=[self.config.target_feature_y])
        self.y = data[self.config.target_feature_y]

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
        """Runs hyperparameter optimization using Bayesian optimization.

        Uses Tree-structured Parzen Estimator (TPE) algorithm to find optimal
        hyperparameters by minimizing specified loss metric on validation set.

        Returns:
            None: Updates self.tuned_hyperparams with optimized parameters.
        """

        def hyperparam_tuning(space):
            """Objective function for hyperparameter optimization.

            Args:
                space (dict): Hyperparameter search space from hyperopt.

            Returns:
                dict: Loss value and status for hyperopt optimization.
            """

            model = XGBRegressor(**space)

            evaluation = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
            model.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)

            # Predictions and scores
            preds = model.predict(self.X_val)
            rmse = mean_squared_error(self.y_val, preds)
            r2 = r2_score(self.y_val, preds)
            mae = np.mean(np.abs(self.y_val - preds))
            print("SCORE:", rmse, mae, r2)

            match self.config.trials_loss_metric:
                case "mae":
                    loss_value = mae
                case "rmse":
                    loss_value = rmse

            return {"loss": loss_value, "status": STATUS_OK, "model": model}

        trials = Trials()
        self.best_trial = fmin(
            fn=hyperparam_tuning,
            space=self.trials_hyperparam_space,
            algo=tpe.suggest,
            max_evals=self.config.trials_param_eval,
            trials=trials,
        )
        self.best_trial = {
            key: int(value) if key in ["max_depth", "max_bin"] else value
            for key, value in self.best_trial.items()
        }
        # Merge hyperparam_space and best_trial to create tuned_hyperparams
        # because best_trial doesn't carry over constant hyperparams when sent through fmin()
        self.tuned_hyperparams = self.trials_hyperparam_space | self.best_trial


class ModelTrain:
    """Orchestrates XGBoost model training with optional iterative evaluation,
    and recursive feature elimination.

    Manages the complete pipeline for XGBoost regression including hyperparameter
    tuning, model training, performance evaluation, and iterative feature removal
    based on feature importance scores.

    Args:
        config (ModelConfig): Configuration object containing pipeline parameters.
        model (ModelOptimisation): XGBoost model instance with tuned hyperparameters.

    Attributes:
        pruned_features_data (pd.DataFrame): Copy of data for feature elimination.
        feature_importance (pd.DataFrame): Feature importance rankings from model.
    """

    def __init__(
        self,
        config: ModelConfig,
        clean: DataCleaner,
        model: ModelOptimisation,
        storage: DataStorage,
    ):
        self.config = config
        self.clean = clean
        self.model = model
        self.storage = storage
        self.feature_importance = pd.DataFrame()

    def recursive_feature_elimination_generator(self):
        """Recursively removes zero-importance features until minimum threshold reached.

        Iteratively identifies features with zero importance scores, removes them from
        the dataset, retrains the model, and repeats until no zero-importance features
        remain or minimum feature threshold is reached.

        Returns:
            None: Modifies self.pruned_features_data and prints elimination progress.
        """

        if not self.config.recursive_trials:
            return

        self.pruned_features_data = (
            self.clean.data.copy()
        )  # Copy for feature elimination

        iter_num = 0
        with alive_bar(unknown="stars", title="Feature elimination") as bar:
            while (
                len(self.pruned_features_data.columns)
                >= self.config.min_features_per_sample
            ):
                iter_num += 1
                prune_col = self.feature_importance.loc[
                    np.isclose(self.feature_importance["Importance"], 0)
                ]["Feature"].tolist()

                if len(prune_col) == 0:
                    break

                self.pruned_features_data = self.pruned_features_data.drop(
                    columns=prune_col
                )
                self.model.set_x_y(self.pruned_features_data)
                self.run_model()

                mae, r2, rmse = self.iteration_stats
                self.storage.store_iteration(
                    iter_num, self.feature_importance, mae, r2, rmse
                )

            self.storage.store_final_model_stats(
                self.feature_importance, self.model.tuned_hyperparams, mae, r2, rmse
            )
            bar()

    def run_model(self):
        """Trains XGBoost model with tuned hyperparameters and caches feature importance.

        Fits the XGBoost regressor using optimized hyperparameters, evaluates performance
        on validation set, and stores feature importance scores for recursive elimination.

        Returns:
            None: Prints model metrics and updates self.feature_importance attribute.
        """

        def show_model_stats(XGBmodel):
            """Prints model performance metrics on validation set"""

            preds = XGBmodel.predict(self.model.X_val)
            rmse = mean_squared_error(self.model.y_val, preds)
            r2 = r2_score(self.model.y_val, preds)
            mae = np.mean(np.abs(self.model.y_val - preds))
            return mae, r2, rmse

        def cache_feature_importance(XGBmodel):
            """Returns DataFrame of features sorted by importance descending.

            Args:
                XGBmodel (XGBRegressor): Trained XGBoost model.

            Returns:
                pd.DataFrame: Feature importance with columns 'Feature' and 'Importance'.
            """

            importance = XGBmodel.feature_importances_
            feature_names = self.model.X_train.columns
            self.feature_importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            ).sort_values("Importance", ascending=False)

        XGBmodel = XGBRegressor(**self.model.tuned_hyperparams)
        XGBmodel.fit(
            self.model.X_train,
            self.model.y_train,
            eval_set=[(self.model.X_val, self.model.y_val)],
            verbose=False,
        )
        self.iteration_stats = mae, r2, rmse = show_model_stats(XGBmodel)
        cache_feature_importance(XGBmodel)


class ModelPipeline:
    """Orchestrates complete XGBoost pipeline from data cleaning to model training.

    Args:
        config (ModelConfig): Configuration object containing pipeline parameters.

    Attributes:
        config (ModelConfig): Configuration object.
        clean (DataCleaner): Data cleaning instance.
        model (ModelOptimisation): Model optimization instance.
        train (ModelTrain): Model training instance.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.clean = DataCleaner(config)
        self.model = ModelOptimisation(config)
        self.storage = DataStorage(config)
        self.train = ModelTrain(config, self.clean, self.model, self.storage)

    def run(self):
        """Runs complete pipeline: cleaning, optimization, training, and feature elimination.

        Returns:
            None: Executes full pipeline and prints results.
        """
        self.clean.read_in_csv()
        self.clean.type_check_and_replace()
        self.clean.drop_user_cols()
        self.clean.target_feature_na_drop()
        self.clean.label_encoding()

        # Model optimization and training
        self.model.set_x_y(self.clean.data)
        self.model.run_trials()
        self.train.run_model()
        self.train.recursive_feature_elimination_generator()
        self.storage.save_final_and_iter_model_stats()
