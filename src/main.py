import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.feature_selection import SelectKBest, f_regression

