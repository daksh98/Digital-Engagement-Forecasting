# Credit given to Selva Prabhakaran for starter code on VAR 
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")

import gc
from pathlib import Path

# ==== DATA PREPROCESSING =====

# download dataset
data = pd.read_pickle('./playerFollowersEng.pkl')

# check the percentage of missing data in each column
per_missing = data.isna().sum()*100/len(data)
per_missing.sort_values(ascending=False)

import datetime 

# check for the period covered by the data (total number of days)
data['dailyDataDate'] = pd.to_datetime(data['dailyDataDate']).dt.normalize()
(data.dailyDataDate.max() - data.dailyDataDate.min()) + datetime.timedelta(days=1)

var_df = data.groupby(['dailyDataDate', 'playerId'])[['target1', 'target2', 'target3', 
                                                      'target4', 'numberOfFollowers']].sum().reset_index()


# plotting data

f, ax = plt.subplots(nrows=5, ncols=1, figsize=(22,20))

sns.lineplot(x=var_df.dailyDataDate, y=var_df.target1, ax=ax[0], color='b')
ax[0].set_title('Target 1 Over Time', fontsize=14)

sns.lineplot(x=var_df.dailyDataDate, y=var_df.target2, ax=ax[1], color='b')
ax[1].set_title('Target 2 Over Time', fontsize=14)

sns.lineplot(x=var_df.dailyDataDate, y=var_df.target3, ax=ax[2], color='b')
ax[2].set_title('Target 3 Over Time', fontsize=14)

sns.lineplot(x=var_df.dailyDataDate, y=var_df.target4, ax=ax[3], color='b')
ax[3].set_title('Target 4 Over Time', fontsize=14)

sns.lineplot(x=var_df.dailyDataDate, y=var_df.numberOfFollowers, ax=ax[4], color='b')
ax[4].set_title('Number of Player Twitter Followers Over Time', fontsize=14)

plt.tight_layout()
plt.show()


# ==== CHECKING CASUALITY ====
from statsmodels.tsa.stattools import grangercausalitytests

'''
Check Granger Causality of all possible combinations of the time series.
Rows are the response variable and the columns are predictors.

==== Params ====
data: pandas dataframe
variables: list containing names of the time series variables
maxlag: the maximum number of lags the time series uses 
'''
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False, maxlag=20):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df  

grangers_causation_matrix(var_df, variables = ['target1', 'target2', 'target3', 'target4', 'numberOfFollowers']) 

# normalising numberOfFollowers between 0 and 100 (similar to target values)
nvar_df = var_df[['numberOfFollowers']]
nFollowers = (nvar_df - nvar_df.min())/(nvar_df.max() - nvar_df.min()) * 100
var_df['numberOfFollowers'] = nFollowers


# ==== TRAIN AND TEST SPLIT ====
index=var_df['dailyDataDate']<'2021-04-01'
df_train, df_test = var_df.loc[index], var_df[~index]

# plot train set data

f, ax = plt.subplots(nrows=5, ncols=1, figsize=(22,20))

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target1, ax=ax[0], color='b')
ax[0].set_title('Target 1 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target2, ax=ax[1], color='b')
ax[1].set_title('Target 2 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target3, ax=ax[2], color='b')
ax[2].set_title('Target 3 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target4, ax=ax[3], color='b')
ax[3].set_title('Target 4 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.numberOfFollowers, ax=ax[4], color='b')
ax[4].set_title('Number of Player Twitter Followers Over Time', fontsize=14)

plt.tight_layout()
plt.show()

# ==== CHECKING STATIONARITY ====
from statsmodels.tsa.stattools import adfuller
'''
Perform Augmented Dickey-Fuller Test for Stationarity

==== Params ====
series: a feature of the time series
signif: the significance level 
name: name of the feature of the time series
'''
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
    
# ADF Test on each column - for training set
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# ==== CHECKING FOR COINTEGRATION =====
from statsmodels.tsa.vector_ar.vecm import coint_johansen

'''
Perform Johansen's Cointegration Test

==== Params ====
df: the dataset
alpha: critical value
'''
def cointegration_test(df, alpha=0.05): 
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df_train[['target1', 'target2', 'target3', 'target4', 'numberOfFollowers']])

# since training set is not stationary due to seasonality, take the first difference

ts_diff = np.diff(df_train.target1)
df_train['target1_1d'] = np.append([0], ts_diff)

ts_diff = np.diff(df_train.target2)
df_train['target2_1d'] = np.append([0], ts_diff)

ts_diff = np.diff(df_train.target3)
df_train['target3_1d'] = np.append([0], ts_diff)

ts_diff = np.diff(df_train.target4)
df_train['target4_1d'] = np.append([0], ts_diff)

ts_diff = np.diff(df_train.numberOfFollowers)
df_train['numberOfFollowers_1d'] = np.append([0], ts_diff)

df_train.set_index('dailyDataDate')

# plot training set after first difference

f, ax = plt.subplots(nrows=5, ncols=1, figsize=(22,20))

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target1_1d, ax=ax[0], color='b')
ax[0].set_title('Target 1 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target2_1d, ax=ax[1], color='b')
ax[1].set_title('Target 2 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target3_1d, ax=ax[2], color='b')
ax[2].set_title('Target 3 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.target4_1d, ax=ax[3], color='b')
ax[3].set_title('Target 4 Over Time', fontsize=14)

sns.lineplot(x=df_train.dailyDataDate, y=df_train.numberOfFollowers_1d, ax=ax[4], color='b')
ax[4].set_title('Number of Player Twitter Followers Over Time', fontsize=14)

plt.tight_layout()
plt.show()

df_train_d = df_train.copy()
df_train_d = df_train_d.drop(['target1', 'target2', 'target3', 'target4', 'numberOfFollowers'], axis=1)

# ADF Test on each column - training set
for name, column in df_train_d.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# ADF Test on each column - test set
for name, column in df_test.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# ==== MODELLING ====
# KFOLD CROSS VALIDATION
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 3

X = df_train_d.dailyDataDate
y = df_train_d[['target1_1d', 'target2_1d', 'target3_1d', 'target4_1d', 'numberOfFollowers_1d']]

folds = TimeSeriesSplit(n_splits=N_SPLITS)

from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

'''
Performs cross validation and returns the MCMAE score

==== Params ====
lag_order: the number of lags the model is fitted on
folds: the number of folds
'''
def cross_validation_score(lag_order, folds):
    
    score_mae = []
    score_mse = []
    
    for train_index, val_index in folds.split(X):
        mae = []
        mse = []

        # prepare training and validation data for this fold
        X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[val_index]
        
        data = pd.concat([X_train, y_train], axis=1)
        data = data.set_index('dailyDataDate')

        # fit model 
        model = VAR(data)
        fitted = model.fit(lag_order)
        
        # prediction 
        y_val_pred = fitted.forecast(fitted.y, steps=X_valid.shape[0])
        y_val_pred = pd.DataFrame(y_val_pred)
            
        
        # calculate metrics
        mae.append(mean_absolute_error(y_valid.target1_1d, y_val_pred.iloc[:, 0]))
        mae.append(mean_absolute_error(y_valid.target2_1d, y_val_pred.iloc[:, 1]))
        mae.append(mean_absolute_error(y_valid.target3_1d, y_val_pred.iloc[:, 2]))
        mae.append(mean_absolute_error(y_valid.target4_1d, y_val_pred.iloc[:, 3]))
        
        # get mean mae of targets
        score_mae.append(np.mean(mae))
        
    return np.mean(score_mae)


'''
Returns the best lag and corresponding scores based on the lowest loss.

==== Params ====
max_lag
folds: number of folds
'''
def get_best_lag(max_lag, folds):
    lag_score = {}
    for lag in range(1, max_lag + 1):
        lag_score[lag] = cross_validation_score(lag, folds)
    
    # find min score
    best_lag = min(lag_score, key=lag_score.get)
    
    return best_lag, lag_score

max_lag = 50
get_best_lag(max_lag, folds)

# ==== TRAINING =====
from statsmodels.tsa.api import VAR

df_train_d = df_train_d.set_index('dailyDataDate')
model = VAR(df_train_d[['target1_1d', 'target2_1d', 'target3_1d', 'target4_1d', 'numberOfFollowers_1d']])

model.select_order(100).summary()

# choosing a lag of 77 based on AIC
fitted = model.fit(77)
fitted.summary()


# ==== CHECK DURBIN-WATSON ====
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(fitted.resid)

for col, val in zip(df_train_d[['target1_1d', 'target2_1d', 'target3_1d', 'target4_1d', 'numberOfFollowers_1d']], out):
    print(col, ':', round(val, 2))


# ==== FORECASTING ====
df_train_fc = df_train_d[['target1_1d', 'target2_1d', 'target3_1d', 'target4_1d', 'numberOfFollowers_1d']]

lag_order = fitted.k_ar

df_train_d.drop(['playerId'], axis=1, inplace=True)
ntest = df_test.shape[0]

predict = fitted.forecast(df_train_d.values[-lag_order:], steps=ntest)
df_forecast = pd.DataFrame(predict, index=df_train_fc.index[-ntest:], columns=df_train_fc.columns)
df_forecast

'''
Revert back the differencing to get the forecasr to original scale

==== Params ====
df_train: training dataset
df_forecast: predictions dataset
'''
def invert_transformation(df_train, df_forecast):
    df_fc = df_forecast.copy()
    df_t = df_train.drop(['playerId', 'target1_1d', 'target2_1d', 'target3_1d', 'target4_1d', 'numberOfFollowers_1d'], axis=1)
    columns = df_t.columns
    for col in columns[1:]:        
        # Roll back 1st Diff
        df_fc[str(col)+'_f'] = df_t[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


df_results = invert_transformation(df_train, df_forecast)
df_results = df_results.loc[:, ['target1_f', 'target2_f', 'target3_f', 'target4_f', 'numberOfFollowers_f']]
df_results

'''
Print summary of forecast accuracy
'''
def forecast_accuracy(forecast, actual):
    mae_li = []
    mse_li = []
    for i in range(len(forecast)): 
        mae = np.abs(forecast[i] - actual[i])
        mae_li.append(mae)
        mse = (forecast[i] - actual[i])**2
        mse_li.append(mse)
    return(np.mean(mae_li), np.mean(mse_li))

df_test = df_test.set_index('dailyDataDate')

print('Forecast Accuracy of: Target 1')
accuracy_prod = forecast_accuracy(df_results.iloc[:, 0], df_test['target1'])
print('mae: ', accuracy_prod[0])
print('mse: ', accuracy_prod[1])
score1 = accuracy_prod[0]
    
print('Forecast Accuracy of: Target 2')
accuracy_prod = forecast_accuracy(df_results.iloc[:, 1], df_test['target2'])
print('mae: ', accuracy_prod[0])
print('mse: ', accuracy_prod[1])
score2 = accuracy_prod[0]
    
print('Forecast Accuracy of: Target 3')
accuracy_prod = forecast_accuracy(df_results.iloc[:, 2], df_test['target3'])
print('mae: ', accuracy_prod[0])
print('mse: ', accuracy_prod[1])
score3 = accuracy_prod[0]
    
print('Forecast Accuracy of: Target 4')
accuracy_prod = forecast_accuracy(df_results.iloc[:, 3], df_test['target4'])
print('mae: ', accuracy_prod[0])
print('mse: ', accuracy_prod[1])
score4 = accuracy_prod[0]

print('score: ', (score1 + score2 + score3 + score4)/4)


