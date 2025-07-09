import pandas as pd
import numpy as np
import warnings
from ta import add_all_ta_features

warnings.filterwarnings("ignore")

# ------------------ 1. Load Data ------------------

file_path = r"C:\Users\AnujShah\nifty_features_data.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df = df[~df.index.duplicated(keep='last')]

custom_start = pd.to_datetime("2017-01-01")
custom_end = pd.to_datetime("2025-07-09")
df = df[(df.index >= custom_start) & (df.index <= custom_end)]

print(f"📅 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------ 1A. Load FII/DII Net Data ------------------

fii_dii_path = r"C:\Users\AnujShah\fii_dii_data.xlsx"
fii_dii = pd.read_excel(fii_dii_path)

# Standardize and prepare
fii_dii['Date'] = pd.to_datetime(fii_dii['Date'])
fii_dii.set_index('Date', inplace=True)
fii_dii = fii_dii.sort_index()

# Rename for convenience
fii_dii.rename(columns={
    'FII - Net Purchase / Sales': 'FII_Net_Equity',
    'DII - Net Purchase / Sale': 'DII_Net_Equity',
    'FII Fut - Net Purchase / Sales': 'FII_Net_Futures',
    'FII Opt - Net Purchase / Sales': 'FII_Net_Options'
}, inplace=True)

# ------------------ 1B. Merge with Main Data ------------------

df = df.merge(
    fii_dii[['FII_Net_Equity', 'DII_Net_Equity', 'FII_Net_Futures', 'FII_Net_Options']],
    left_index=True, right_index=True, how='left'
)

# ------------------ 1C. FII/DII Feature Engineering ------------------

# Raw Net Flows
df['FII_DII_Net'] = df['FII_Net_Equity'] + df['DII_Net_Equity']
df['FII_minus_DII'] = df['FII_Net_Equity'] - df['DII_Net_Equity']

# 3-day and 5-day rolling means
df['FII_rolling_3d'] = df['FII_Net_Equity'].rolling(3).mean()
df['DII_rolling_3d'] = df['DII_Net_Equity'].rolling(3).mean()
df['FII_rolling_5d'] = df['FII_Net_Equity'].rolling(5).mean()
df['DII_rolling_5d'] = df['DII_Net_Equity'].rolling(5).mean()
df['FII_DII_rolling_5d'] = df['FII_DII_Net'].rolling(5).mean()

# Momentum of flows
df['FII_mom_3d'] = df['FII_Net_Equity'] - df['FII_Net_Equity'].shift(3)
df['DII_mom_3d'] = df['DII_Net_Equity'] - df['DII_Net_Equity'].shift(3)

# Volatility of flows
df['FII_std_5d'] = df['FII_Net_Equity'].rolling(5).std()
df['DII_std_5d'] = df['DII_Net_Equity'].rolling(5).std()

# Net flow percent change
df['FII_pct_change_1d'] = df['FII_Net_Equity'].pct_change()
df['DII_pct_change_1d'] = df['DII_Net_Equity'].pct_change()

# Ratios
df['FII_DII_Ratio'] = df['FII_Net_Equity'] / df['DII_Net_Equity'].replace(0, np.nan)
df['FII_to_Total_Ratio'] = df['FII_Net_Equity'] / df['FII_DII_Net'].replace(0, np.nan)

# Smoothed flows
df['FII_EMA_5'] = df['FII_Net_Equity'].ewm(span=5).mean()
df['DII_EMA_5'] = df['DII_Net_Equity'].ewm(span=5).mean()

# ------------------ New: FII Futures & Options Feature Engineering ------------------

# Rolling averages
df['FII_Fut_rolling_3d'] = df['FII_Net_Futures'].rolling(3).mean()
df['FII_Opt_rolling_3d'] = df['FII_Net_Options'].rolling(3).mean()

# Momentum
df['FII_Fut_mom_3d'] = df['FII_Net_Futures'] - df['FII_Net_Futures'].shift(3)
df['FII_Opt_mom_3d'] = df['FII_Net_Options'] - df['FII_Net_Options'].shift(3)

# Volatility
df['FII_Fut_std_5d'] = df['FII_Net_Futures'].rolling(5).std()
df['FII_Opt_std_5d'] = df['FII_Net_Options'].rolling(5).std()

# EMA
df['FII_Fut_EMA_5'] = df['FII_Net_Futures'].ewm(span=5).mean()
df['FII_Opt_EMA_5'] = df['FII_Net_Options'].ewm(span=5).mean()

# ------------------ New: Buy/Sell/Neutral Status ------------------

def get_status(x):
    if x > 500:
        return "Buy"
    elif x < -500:
        return "Sell"
    else:
        return "Neutral"

df['FII_Cash_Status'] = df['FII_Net_Equity'].apply(get_status)
df['DII_Cash_Status'] = df['DII_Net_Equity'].apply(get_status)
df['FII_Fut_Status'] = df['FII_Net_Futures'].apply(get_status)
df['FII_Opt_Status'] = df['FII_Net_Options'].apply(get_status)

status_map = {'Sell': -1, 'Neutral': 0, 'Buy': 1}
df['FII_Cash_Status_Num'] = df['FII_Cash_Status'].map(status_map)
df['DII_Cash_Status_Num'] = df['DII_Cash_Status'].map(status_map)
df['FII_Fut_Status_Num'] = df['FII_Fut_Status'].map(status_map)
df['FII_Opt_Status_Num'] = df['FII_Opt_Status'].map(status_map)

df.drop(columns=[
    'FII_Cash_Status', 'DII_Cash_Status',
    'FII_Fut_Status', 'FII_Opt_Status'
], inplace=True)

# ------------------ 2. Feature Engineering ------------------

tickers = ["NIFTY", "BANKNIFTY", "USDINR", "MIDCAP"]
for name in tickers:
    if f"{name}_Close" in df.columns:
        df[f"{name}_ret1d"] = df[f"{name}_Close"].pct_change()
        df[f"{name}_ret2d"] = df[f"{name}_Close"].pct_change(2)

df = add_all_ta_features(
    df,
    open="NIFTY_Open", high="NIFTY_High", low="NIFTY_Low",
    close="NIFTY_Close", volume="NIFTY_Volume",
    fillna=True
)

for lag in [1, 2, 3]:
    df[f'NIFTY_lag{lag}'] = df['NIFTY_Close'].shift(lag)

for window in [5, 10, 20]:
    df[f'NIFTY_MA_{window}'] = df['NIFTY_Close'].rolling(window).mean()
df['NIFTY_MA_Cross'] = df['NIFTY_MA_5'] - df['NIFTY_MA_20']

df['NIFTY_EMA_5'] = df['NIFTY_Close'].ewm(span=5).mean()
df['NIFTY_EMA_20'] = df['NIFTY_Close'].ewm(span=20).mean()
df['NIFTY_EMA_diff'] = df['NIFTY_EMA_5'] - df['NIFTY_EMA_20']

df['NIFTY_ret3d'] = df['NIFTY_Close'].pct_change(3)
df['NIFTY_ret5d'] = df['NIFTY_Close'].pct_change(5)
df['NIFTY_ret10d'] = df['NIFTY_Close'].pct_change(10)
df['NIFTY_Momentum_10'] = df['NIFTY_Close'] - df['NIFTY_Close'].shift(10)

df['NIFTY_STD_5'] = df['NIFTY_Close'].rolling(5).std()
df['RET_std_5d'] = df['NIFTY_ret1d'].rolling(5).std()
df['ATR_14'] = (df['NIFTY_High'] - df['NIFTY_Low']).rolling(14).mean()
df['Bollinger_Upper'] = df['NIFTY_MA_20'] + 2 * df['NIFTY_STD_5']
df['Bollinger_Lower'] = df['NIFTY_MA_20'] - 2 * df['NIFTY_STD_5']
df['Bollinger_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']

df['NIFTY_Volume_Change_3d'] = df['NIFTY_Volume'].pct_change(3)
df['NIFTY_Volume_Change_5d'] = df['NIFTY_Volume'].pct_change(5)
df['NIFTY_Volume_Change'] = df['NIFTY_Volume'] / df['NIFTY_Volume'].shift(1)

obv = [0]
for i in range(1, len(df)):
    if df['NIFTY_Close'].iloc[i] > df['NIFTY_Close'].iloc[i - 1]:
        obv.append(obv[-1] + df['NIFTY_Volume'].iloc[i])
    elif df['NIFTY_Close'].iloc[i] < df['NIFTY_Close'].iloc[i - 1]:
        obv.append(obv[-1] - df['NIFTY_Volume'].iloc[i])
    else:
        obv.append(obv[-1])
df['OBV'] = obv

df['VWAP'] = (
    (df['NIFTY_High'] * df['NIFTY_Volume'] +
     df['NIFTY_Low'] * df['NIFTY_Volume'] +
     df['NIFTY_Close'] * df['NIFTY_Volume']) / (3 * df['NIFTY_Volume'])
)
df['VWAP_diff'] = df['NIFTY_Close'] - df['VWAP']

df['NIFTY_vs_BANKNIFTY'] = df['NIFTY_ret1d'] - df['BANKNIFTY_ret1d']
df['NIFTY_REL'] = df['NIFTY_Close'] / df['BANKNIFTY_Close']

df['DayOfWeek'] = df.index.dayofweek

# ------------------ 6. ADDITIONAL ADVANCED FEATURES ------------------

df['NIFTY_log_ret1d'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(1))
df['NIFTY_log_ret3d'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(3))

df['NIFTY_EMA_ratio_5_20'] = df['NIFTY_EMA_5'] / df['NIFTY_EMA_20']

df['NIFTY_ret1d_std5'] = (
    (df['NIFTY_ret1d'] - df['NIFTY_ret1d'].rolling(5).mean()) /
    df['NIFTY_ret1d'].rolling(5).std()
)

df['NIFTY_overnight_gap'] = df['NIFTY_Open'] - df['NIFTY_Close'].shift(1)
df['NIFTY_overnight_gap_pct'] = df['NIFTY_overnight_gap'] / df['NIFTY_Close'].shift(1)

vol_rolling = df['NIFTY_ret1d'].rolling(20).std()
low_thresh = vol_rolling.rolling(252, min_periods=20).quantile(0.33)
high_thresh = vol_rolling.rolling(252, min_periods=20).quantile(0.66)
df['NIFTY_vol_regime'] = 'medium'
df.loc[vol_rolling <= low_thresh, 'NIFTY_vol_regime'] = 'low'
df.loc[vol_rolling >= high_thresh, 'NIFTY_vol_regime'] = 'high'

# --------------- INTERACTION FEATURES ---------------

# Encode regime as numeric for interactions
regime_map = {'low': 0, 'medium': 1, 'high': 2}
df['NIFTY_vol_regime_encoded'] = df['NIFTY_vol_regime'].map(regime_map)

# Example interaction features:
# 1. NIFTY return × FII flow
df['NIFTY_ret1d_x_FII'] = df['NIFTY_ret1d'] * df['FII_Net_Equity']
df['NIFTY_ret1d_x_DII'] = df['NIFTY_ret1d'] * df['DII_Net_Equity']
df['NIFTY_ret1d_x_FII_DII'] = df['NIFTY_ret1d'] * df['FII_DII_Net']

# 2. Regime × indicator/flows
df['regime_x_rsi'] = df['NIFTY_vol_regime_encoded'] * df.get('momentum_rsi', np.nan)
df['regime_x_macd'] = df['NIFTY_vol_regime_encoded'] * df.get('trend_macd', np.nan)
df['regime_x_fii'] = df['NIFTY_vol_regime_encoded'] * df['FII_Net_Equity']
df['regime_x_dii'] = df['NIFTY_vol_regime_encoded'] * df['DII_Net_Equity']

# 3. Regime × NIFTY return
df['regime_x_ret1d'] = df['NIFTY_vol_regime_encoded'] * df['NIFTY_ret1d']

# 4. NIFTY return × MACD
df['NIFTY_ret1d_x_macd'] = df['NIFTY_ret1d'] * df.get('trend_macd', np.nan)

# 5. FII Flow × RSI
df['FII_x_rsi'] = df['FII_Net_Equity'] * df.get('momentum_rsi', np.nan)

# 6. DII Flow × MACD
df['DII_x_macd'] = df['DII_Net_Equity'] * df.get('trend_macd', np.nan)

# 7. Return × (FII - DII)
df['NIFTY_ret1d_x_FII_minus_DII'] = df['NIFTY_ret1d'] * df['FII_minus_DII']

# 8. FII × DII (joint effect)
df['FII_x_DII'] = df['FII_Net_Equity'] * df['DII_Net_Equity']

# ------------------ 3. Targets ------------------

df['Target_ret'] = df['NIFTY_Close'].pct_change(periods=1).shift(-1).clip(-0.04, 0.04)
df['Target_dir'] = (df['Target_ret'] > 0).astype(int)

print(f"📊 Target statistics:")
print(f"Target_ret: {df['Target_ret'].describe()}")
print(f"Target_dir distribution: {df['Target_dir'].value_counts()}")

# ------------------ 4. Clean NaNs from Features ONLY ------------------

target_cols = ['Target_ret', 'Target_dir']
feature_cols = [col for col in df.columns if col not in target_cols]

df_features = df[feature_cols].dropna()
df = df.loc[df_features.index]  # Align target values with feature-available dates

# ------------------ 5. Final Checks ------------------

print(f"✅ Final usable dataset: {df.shape}")
print(f"📅 Date Range:\nStart: {df.index.min().date()}  End: {df.index.max().date()}")
print(f"🧪 Last available row (for prediction): {df.tail(1).index[0].date()}")


from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# --- Feature engineering and target creation assumed done above ---
if 'NIFTY_vol_regime' in df.columns:
    df['NIFTY_vol_regime_encoded'] = df['NIFTY_vol_regime'].map({'low': 0, 'medium': 1, 'high': 2})

df['Target_ret'] = df['NIFTY_Close'].pct_change(periods=1).shift(-1).clip(-0.04, 0.04)
df['Target_dir'] = (df['Target_ret'] > 0).astype(int)

target_cols = ['Target_ret', 'Target_dir']
features = [
    col for col in df.columns
    if col not in target_cols and col != 'NIFTY_vol_regime'
]
X = df[features]
y_reg = df['Target_ret']
y_clf = df['Target_dir']

X = X.replace([np.inf, -np.inf], np.nan).dropna()
y_reg = y_reg.loc[X.index]
y_clf = y_clf.loc[X.index]

valid_idx = y_reg.dropna().index
X = X.loc[valid_idx]
y_reg = y_reg.loc[valid_idx]
y_clf = y_clf.loc[valid_idx]

# --- 🔧 1. Separate Feature Selection for Classification ---
fs_reg = RandomForestRegressor(n_estimators=100, random_state=42)
fs_reg.fit(X, y_reg)
selector_reg = SelectFromModel(fs_reg, threshold="median", prefit=True)
X_selected = selector_reg.transform(X)

fs_clf = RandomForestClassifier(n_estimators=100, random_state=42)
fs_clf.fit(X, y_clf)
selector_clf = SelectFromModel(fs_clf, threshold="median", prefit=True)
X_selected_clf = selector_clf.transform(X)

# --- Scaling (without data leakage) ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit only on data till T-1 (excluding last row)
scaler_X.fit(X_selected[:-1])
scaler_y.fit(y_reg.values[:-1].reshape(-1, 1))

# Now transform all (including the last row, safely)
X_scaled = scaler_X.transform(X_selected)
y_reg_scaled = scaler_y.transform(y_reg.values.reshape(-1, 1)).ravel()

# Latest (T) row for prediction (already selected earlier)
latest_data_selected = selector_reg.transform(X.iloc[[-1]])
latest_data_selected_clf = selector_clf.transform(X.iloc[[-1]])
latest_data_scaled = scaler_X.transform(latest_data_selected)

# --- Define models ---
reg_models = {
    'XGB_Reg': XGBRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=42),
    'RF_Reg': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR_Reg': SVR(),
    'MLP_Reg': MLPRegressor(
    hidden_layer_sizes=(10,),
    alpha=0.01,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    learning_rate_init=0.0005,
    activation='tanh'
),
    'LGBM_Reg': LGBMRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=42, verbose=-1),
    'CatBoost_Reg': CatBoostRegressor(iterations=50, learning_rate=0.01, depth=5, random_state=42, verbose=0)
}

clf_models = {
    'Log_Clf': LogisticRegression(),
    'RF_Clf': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=2, random_state=42),
    'SVC_Clf': CalibratedClassifierCV(SVC(probability=True)),  # 🔧 5. Calibration
    'MLP_Clf': CalibratedClassifierCV(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    'LGBM_Clf': LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1),
    'CatBoost_Clf': CatBoostClassifier(iterations=50, depth=5, learning_rate=0.01, random_state=42, verbose=0)
}

# --- 🔧 4. Regression MAE with Inverse-Transform ---
tscv = TimeSeriesSplit(n_splits=5)
print("\n📊 Cross-validated MAE for Regression Models:")
for name, model in reg_models.items():
    mae_list = []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_reg_scaled[train_idx], y_reg_scaled[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(
            scaler_y.inverse_transform(y_test.reshape(-1, 1)),
            scaler_y.inverse_transform(np.array(preds).reshape(-1, 1))
        )
        mae_list.append(mae)
    print(f"{name}: Mean MAE = {np.mean(mae_list):.6f}")

# --- Latest Predictions ---
latest_preds = {}

# ➤ Regression
for name, model in reg_models.items():
    try:
        model.fit(X_scaled[:-1], y_reg_scaled[:-1])
        pred_scaled = model.predict(latest_data_scaled)
        pred = scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).ravel()[0]

        latest_preds[name] = {
            'Prediction': round(pred, 6),
            'Type': 'Regression',
            'Direction': 'UP' if pred > 0 else 'DOWN'
        }

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

# ➤ Classification
for name, model in clf_models.items():
    try:
        model.fit(X_selected_clf[:-1], y_clf.values[:-1])
        pred = model.predict(latest_data_selected_clf)[0]
        pred_dir = 'UP' if pred == 1 else 'DOWN'

        entry = {
            'Prediction': int(pred),
            'Type': 'Classification',
            'Direction': pred_dir
        }

        # For CatBoost/LGBM, predict_proba may throw if binary_classification is not properly set, so catch exception
        prob = None
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(latest_data_selected_clf)[0][1]
        except Exception:
            prob = None
        if prob is not None:
            entry['Probability'] = round(prob, 3)

        latest_preds[name] = entry

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

# ✅ Display Predictions
print("\n📈 Latest Model Predictions:")
for model_name, pred in latest_preds.items():
    print(f"{model_name}: {pred}")

# Format regression results to percentage
def format_prediction(val, model_type):
    if model_type == 'Regression':
        return f"{val * 100:.2f}%"
    return val

summary_df = pd.DataFrame(latest_preds).T
summary_df['Prediction'] = summary_df.apply(lambda row: format_prediction(row['Prediction'], row['Type']), axis=1)

# Optional: Add emojis to Direction
summary_df['Direction'] = summary_df['Direction'].map({'UP': '📈 UP', 'DOWN': '📉 DOWN'})

# Optional: Clean column order
summary_df = summary_df[['Type', 'Prediction', 'Direction'] + 
                        [col for col in summary_df.columns if col not in ['Type', 'Prediction', 'Direction']]]

print("\n📊 Final Prediction Summary:")
print(summary_df)


# Format regression results to percentage
def format_prediction(val, model_type):
    if model_type == 'Regression':
        return f"{val * 100:.2f}%"
    return val

# Build DataFrame
summary_df = pd.DataFrame(latest_preds).T
summary_df['Prediction'] = summary_df.apply(lambda row: format_prediction(row['Prediction'], row['Type']), axis=1)

# Display prediction table
print("\n📊 Final Prediction Summary:")
print(summary_df)

# --- 🧠 Add Ensemble Logic ---

# 1. Regression: Average return prediction
regression_raw_preds = {
    k: v['Prediction'] for k, v in latest_preds.items() if v['Type'] == 'Regression'
}
regression_avg = np.mean(list(regression_raw_preds.values()))
regression_direction = "UP" if regression_avg > 0 else "DOWN"

# 2. Classification: Majority vote
classification_preds = {
    k: v['Prediction'] for k, v in latest_preds.items() if v['Type'] == 'Classification'
}
clf_votes = list(classification_preds.values())
clf_majority = int(np.round(np.mean(clf_votes)))  # 0 or 1
clf_direction = "UP" if clf_majority == 1 else "DOWN"

# 3. Display ensemble summary
print("\n🧮 Ensemble Summary:")
print(f"📌 Average Regression Return: {regression_avg:.2%} → Direction: {regression_direction}")
print(f"🗳️ Classification Majority Vote: {clf_majority} → Direction: {clf_direction}")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np

# --- Base Regression Models ---
base_reg_models = {
    'XGB_Reg': XGBRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=42),
    'RF_Reg': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR_Reg': SVR(),
    'MLP_Reg': MLPRegressor(
    hidden_layer_sizes=(10,),
    alpha=0.01,
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    learning_rate_init=0.0005,
    activation='tanh'
),
    'LGBM_Reg': LGBMRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=42, verbose=-1),
    'CatBoost_Reg': CatBoostRegressor(iterations=50, learning_rate=0.01, depth=5, random_state=42, verbose=0)
}

tscv = TimeSeriesSplit(n_splits=5)
n_models = len(base_reg_models)
n_samples = X_scaled.shape[0]

# --- Step 1: Out-of-Fold Meta Features ---
meta_features = np.full((n_samples, n_models), np.nan)
meta_target = y_reg_scaled.copy()

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
    for i, (name, model) in enumerate(base_reg_models.items()):
        model_clone = clone(model)
        model_clone.fit(X_scaled[train_idx], y_reg_scaled[train_idx])
        meta_features[test_idx, i] = model_clone.predict(X_scaled[test_idx])

# --- Step 2: Clean Meta Features ---
valid_rows = ~np.isnan(meta_features).any(axis=1)
meta_features_clean = meta_features[valid_rows]
meta_target_clean = meta_target[valid_rows]

# --- Step 3: Scale Meta Features ---
scaler_meta = StandardScaler()
meta_features_scaled = scaler_meta.fit_transform(meta_features_clean)

# --- Step 4: Train Meta-Learner ---
meta_regressor = Ridge(alpha=0.1)
meta_regressor.fit(meta_features_scaled, meta_target_clean)

# --- Step 5: Compute OOF MAEs (for weights) ---
meta_oof_pred = meta_regressor.predict(meta_features_scaled)
meta_mae = mean_absolute_error(meta_target_clean, meta_oof_pred)

reg_avg_oof = np.nanmean(meta_features, axis=1)[valid_rows]
reg_mae = mean_absolute_error(meta_target_clean, reg_avg_oof)

# --- Step 6: Predict Latest from Base Models ---
latest_preds_reg = []
for name, model in base_reg_models.items():
    model.fit(X_scaled[:-1], y_reg_scaled[:-1])  # Exclude latest row
    pred = model.predict(latest_data_scaled)[0]
    latest_preds_reg.append(pred)

# --- Step 7: Predict from Meta Learner ---
latest_meta_input = scaler_meta.transform([latest_preds_reg])
final_meta_scaled = meta_regressor.predict(latest_meta_input)[0]

# Ensure proper shape for inverse_transform
final_meta_pred = scaler_y.inverse_transform(np.array([[final_meta_scaled]]))[0][0]

# --- Step 8: Predict from Base Model Average ---
regression_avg_scaled = np.mean(latest_preds_reg)

# Ensure proper shape for inverse_transform
regression_avg_inv = scaler_y.inverse_transform(np.array([[regression_avg_scaled]]))[0][0]


# --- Step 8: Predict from Base Model Average ---
regression_avg_scaled = np.mean(latest_preds_reg)
regression_avg_inv = scaler_y.inverse_transform([[regression_avg_scaled]])[0][0]

# --- Step 9: Weighted Blending (MAE-based) ---
epsilon = 1e-6  # Prevent division by zero
w_meta = 1 / (meta_mae + epsilon)
w_avg = 1 / (reg_mae + epsilon)
total_weight = w_meta + w_avg

blended_pred = (w_meta * final_meta_pred + w_avg * regression_avg_inv) / total_weight

# --- Final Output ---
print("\n📈 Meta-Learner Ensemble Output:")
print(f"📌 Meta Prediction      : {final_meta_pred:.4%}")
print(f"📌 Base Avg Prediction  : {regression_avg_inv:.4%}")
print(f"📌 Weighted Blend       : {blended_pred:.4%}")
print(f"📌 Final Direction      : {'UP' if blended_pred > 0 else 'DOWN'}")


