from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


_PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _PROJ_ROOT / "data" / "model_data.parquet"

# Training Variables
N_SPLITS = 5
TEST_PERCENTAGE = 0.2

def get_train_range(start_str, latest_str, test_percentage=TEST_PERCENTAGE):
    return latest_str - (latest_str - start_str) * test_percentage 

def load_data(data=DATA_PATH):
    df = pd.read_parquet(data)
    df = df.sort_index()

    test_date = get_train_range(df.index[0], df.index[-1])
    test_df = df[df.index >= test_date].drop(columns='y')
    
    train_val_df = df[df.index < test_date]
    X_train_val = train_val_df.drop(columns='y')
    Y_train_val = train_val_df['y']

    return X_train_val, Y_train_val, test_df

def train_model(X_train_val, Y_train_val):
    
    model = XGBClassifier(verbosity=2,
                          learning_rate=0.01,
                          n_estimators=99999,
                          max_depth=7,
                          early_stopping_rounds=50,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          reg_lambda=1,
                          reg_alpha=0,
                          objective='binary:logistic',
                          eval_metric='logloss'
                          )
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=5)
    
    for train_index, val_index in tscv.split(X_train_val):       
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = Y_train_val.iloc[train_index], Y_train_val.iloc[val_index]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    

if __name__ == "__main__":
    X_train_val, Y_train_val, test_df = load_data()
    train_model(X_train_val, Y_train_val)