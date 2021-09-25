#!/usr/bin/env python
"""AutoRegressiveNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DYKt9gZFTE_5gEBzq5LsdM5pkb8KobBH
"""

import gc
import os
import sys
import warnings
from pathlib import Path



import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import TimeSeriesSplit

import pickle

warnings.simplefilter("ignore")


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

"""### Methodolgy

- We first take a look take a look our data and load the relevent files

- We then develop a neural network where the features are lagged target values (up to 5 day)
    - to develop this we first build the lag data base

    - we then train and test a base model using this lagged data , using K - fold cross validation, to obtain some preliminary results

    - Hyper parameter tuning is then conducted for drop out rate, hidden layers and learning rate for this model

    - then we retrain model using those hyper parameters

    - then obtain a loss score on a held out test set

- Next we move on to building a nerual network where the features are lagged target values up to 20 days
    - we build this model using the optmial hyper paramters found earlier

    - train the model using K- fold cross validation and obtain a loss score on a held out test set

- Finally a neural network is constructed where the features are lagged target values of upto 5 days and key Player box scores
    - we build this model using the optmial hyper paramters found earlier

    - train the model using K- fold cross validation and obtain a loss score on a held out test set

### Note on Data
 - intially data was provided in the form of various csv files , train csv contained various nesting of data as json files.
 - this were unested and converted to pickle files , which have been used through out the notebook

 - please set path to location of a file before use

 - data files used are :
     - nextDayPlayerEngagement_train.pkl
     - players.csv
     - rosters_train.pkl
     - playerBoxScores_train.pkl
"""

# %%capture

# !pip install pandarallel

# import gc

# import numpy as np
# import pandas as pd
# from pathlib import Path

# from pandarallel import pandarallel
# pandarallel.initialize()

# BASE_DIR = Path('../input/mlb-player-digital-engagement-forecasting')
# train = pd.read_csv(BASE_DIR / 'train.csv')

# null = np.nan
# true = True
# false = False

# for col in train.columns:

#     if col == 'date': continue

#     _index = train[col].notnull()
#     train.loc[_index, col] = train.loc[_index, col].parallel_apply(lambda x: eval(x))

#     outputs = []
#     for index, date, record in train.loc[_index, ['date', col]].itertuples():
#         _df = pd.DataFrame(record)
#         _df['index'] = index
#         _df['date'] = date
#         outputs.append(_df)

#     outputs = pd.concat(outputs).reset_index(drop=True)

#     outputs.to_csv(f'{col}_train.csv', index=False)
#     outputs.to_pickle(f'{col}_train.pkl')

#     del outputs
#     del train[col]
#     gc.collect()

"""##### Checking / Collecting our data files  """

path_target = "/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl"
targets = pd.read_pickle(path)

display(targets)

path_bs = '/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/playerBoxScores_train.pkl'
playerBoxScores_train = pd.read_pickle(path_bs)

path_players = '/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/players.csv'
players = pd.read_csv(path_players)

rosters_path = '/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/rosters_train.pkl'
rosters = pd.read_pickle(rosters_path)



"""##### Setting up data for NN """

pl_eng_w_scores = pd.merge(targets,playerBoxScores_train,on=['date','playerId'],how = 'inner')
display(pl_eng_w_scores)

# we want to get birthCountry, primaryPositionName, rosterTeamId, rosterStatus for each player

players_cols = ['playerId', 'primaryPositionName', 'birthCountry']
rosters_cols = ['playerId', 'teamId', 'status', 'date']

position_birth = players[players_cols]
rosters_small = rosters[rosters_cols]
categorical_info = pd.merge(rosters_small,position_birth,on=['playerId'],how = 'left')

pl_eng_w_scores_c = pd.merge(pl_eng_w_scores,categorical_info,on=['playerId' , 'date' ,'teamId'],how = 'left').reset_index()

display(pl_eng_w_scores_c)

del pl_eng_w_scores_c

display(targets)
#resetting
#targets = pd.read_pickle("/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl")

"""### Autoregressive Neural Network - 5 lags

#### Base Model
- we build baseline model first , i.e. run all the cells until hyper paramter tuning , to get a 'feel' for what the model would be
- then we tune hyper parameters
"""

#targets = pd.read_pickle("/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl")

def reset():
    path_target = "/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl"
    targets = pd.read_pickle(path)
    #data differencing
    testing = targets["target1"]
    diff = testing.diff()
    targets["t1_diff"] = diff

    testing = targets["target2"]
    diff = testing.diff()
    targets["t2_diff"] = diff

    testing = targets["target3"]
    diff = testing.diff()
    targets["t3_diff"] = diff

    testing = targets["target4"]
    diff = testing.diff()
    targets["t4_diff"] = diff

    return targets

targets = reset()
display(targets)

#### we now run the code cells in order and build a base line model , which we later tune

from datetime import timedelta

trgt_lags = targets
#display(trgt_lags)

# Functions below will iteratively adds lagged version of targets to the feature set we build.

# using the next day prediction targets as the features in our model by changing the date to one day back
# then we can build lags using this
def day_back(trgt_lags):
    trgt_lags["engagementMetricsDate"] = pd.to_datetime(trgt_lags["engagementMetricsDate"])
    trgt_lags["engagementMetricsDate"] = trgt_lags["engagementMetricsDate"] + timedelta(days=-1)

day_back(trgt_lags)

display(trgt_lags) #checking if our dates did get pushed back

trgt_cols = ["t1_diff","t2_diff","t3_diff","t4_diff"]

def train_lag(df, lag):
    #get copy of current data frame , with targets values
    dp = df[["playerId","engagementMetricsDate"]+trgt_cols].copy()
    # set date to 'lag' many days AHEAD
    dp["engagementMetricsDate"]  = dp["engagementMetricsDate"] + timedelta(days=lag)
    # now when you merge the data frame dp on to the original data frame youll be creating features  that are lagged values
    df = df.merge(dp, on=["playerId", "engagementMetricsDate"], suffixes=["",f"_{lag}"], how="left")
    return df

"""**If anything is changed in the code you must re intialise the trgt_lags variable and consequently train_lag**"""

#building our feature set ...

def create_lags(max_lag, offset, trgt_lags):
    MAX_LAG = max_lag
    OFFSET = offset
    # get list of lags
    LAGS = list(range(OFFSET, MAX_LAG + OFFSET))

    # naming our features
    FECOLS = [f"{col}_{lag}" for lag in reversed(LAGS) for col in trgt_cols]

    # iteratively building our feature set
    for lag in LAGS:
        trgt_lags = train_lag(trgt_lags, lag=lag)
        #print('train---')
        #display(trgt_lags)
        gc.collect()
    return trgt_lags, FECOLS

# max_lag found from data exploration , but using trail and error MAX_LAG of 20 produced the best results
# offset is how many days before you want to start the lags

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #creating our lags data set
# def create_lags_dataset(trgt_lags , LAG, OFFSET):
#     trgt_lags, FECOLS = create_lags(LAG, OFFSET, trgt_lags)
#     trgt_lags = trgt_lags.sort_values(by=["playerId", "engagementMetricsDate"])
#     print('train sorted---')
#     display(trgt_lags)
#     trgt_lags = trgt_lags.dropna()
#     print(trgt_lags.shape)
#     gc.collect()
#
#     return trgt_lags, FECOLS

trgt_lags, FECOLS = create_lags_dataset(trgt_lags , 5, 1)

display(trgt_lags.columns)

def create_train_test(trgt_lags):
    trgt_lags.fillna(-1,inplace = True)

    sample_y = trgt_lags[['t1_diff','t2_diff', 't3_diff', 't4_diff']]
    sample_y = sample_y.reset_index(drop=True)
    sample_X = trgt_lags[FECOLS]
    sample_X = sample_X.reset_index(drop=True)

    # display(sample_X , sample_X.columns)
    # display(sample_y)


    from sklearn.model_selection import train_test_split


    X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2, shuffle=False)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_train_test(trgt_lags)

display(X_train , y_train)

# dropout in hidden layers with weight constraint
def create_model(input_dim , HIDDEN , ACTIVATION, DROPOUT_RATE,LEARNING_RATE):
    #create model
    model = keras.Sequential([
        #keras.Input(shape=(input_dim,)),
        layers.Dense(HIDDEN, activation=ACTIVATION),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(HIDDEN, activation=ACTIVATION),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(4),#output
    ])
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

"""##### Training nueral network with Cross validation """

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def run_ARNN(X_train, X_test, y_train, y_test, hidden, dropuout_rate, learning_rate):
    # Define the model architecture
    HIDDEN = hidden
    ACTIVATION = 'relu'
    DROPOUT_RATE = dropuout_rate
    LEARNING_RATE = learning_rate
    BATCH_SIZE = 5000 # increased batch size as 2.5mill+ rows
    OUTPUTS = 4

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    tscv = TimeSeriesSplit(n_splits=10)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train_index, val_index in tscv.split(X_train):
        input_dim = X_train.shape[0]
        model = create_model(input_dim , HIDDEN , ACTIVATION, DROPOUT_RATE,LEARNING_RATE)
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(
        X_train.iloc[train_index], y_train.iloc[train_index], #validation_data=(X_valid, y_valid)
        batch_size=BATCH_SIZE,
        epochs=15,
        verbose=1)


        # Generate generalization metrics
        scores = model.evaluate(X_train.iloc[val_index], y_train.iloc[val_index], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1
    #after training we evaulte on test set


    return acc_per_fold, loss_per_fold, model

# Hyperparameters
# HIDDEN = 1024
# ACTIVATION = 'relu'  # could try elu, gelu, swish
# DROPOUT_RATE = 0.5
# LEARNING_RATE = 1e-2
# BATCH_SIZE = 32


acc_per_fold, loss_per_fold , model = run_ARNN(X_train, X_test, y_train, y_test, 50, 0.2, 1e-2)

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#results on test set
predictions = model.predict(X_test)
display(predictions)
scores = model.evaluate(X_test, y_test, verbose=0)
print(f'Score for test {model.metrics_names[0]} of {scores[0]}')



#targets = pd.read_pickle("/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl")

"""#### Hyper paramter methodology

- Once baseline model has been run
- we go through each tuning section dropout , nodes and learning rate
    - once you have run one section update the base line function, by updating the relevant variable with the best results obtained
    - base line function :
       - run_ARNN(X_train, X_test, y_train, y_test, hidden, dropuout_rate, learning_rate):
       - variables to update inside function
       - HIDDEN = hidden
       - DROPOUT_RATE = dropuout_rate
       - LEARNING_RATE = learning_rate
- hence you will iteratively building the best model
- we can then go through and build the rest of the models with these optimised paramters (setting the varaibles in the run_ARNN function) as its the same model just with slightly different data
- this method was chose to minimize training time and still give adequt results as opposed to keras tunner

### Hyper paramter tuning with 5 lags

#### Testing different dropout layers
"""

del trgt_lags, X_train, X_test, y_train, y_test

x_dp = []
y_dp = []
LAG = 5
OFFSET = 1
def get_best_dp():
    for dp in np.arange(0, 1.0, 0.1):
        print(f"hidden is --- {dp}")

        #get targets
        targets = reset()
        trgt_lags = targets
        day_back(trgt_lags)

        #get target lags data set
        trgt_lags, FECOLS = create_lags_dataset(trgt_lags ,LAG ,OFFSET)

        display(trgt_lags)
        #get sample set
        X_train, X_test, y_train, y_test = create_train_test(trgt_lags)

        #get model results acc_per_fold, loss_per_fold , model                           hidden, dropuout_rate, learning_rate
        acc_per_fold, loss_per_fold , model = run_ARNN(X_train, X_test, y_train, y_test, 50, dp, 1e-2)

        #find avg loss from each fold
        loss = np.mean(loss_per_fold)
        #append to arrays
        x_dp.append(dp)
        y_dp.append(round(loss, 3))

        #delete all data sets to reset
        del trgt_lags, X_train, X_test, y_train, y_test, acc_per_fold, loss_per_fold , model

#get_best_dp() -- run this to get the arrays
index = y_dp.index(min(y_dp))
best_dp = x_dp[index]
print(best_dp)

del trgt_lags, X_train, X_test, y_train, y_test, acc_per_fold, loss_per_fold , model

"""#### Testing number of Hidden Nodes"""

x_hid = []
y_hid = []

LAG = 5
OFFSET = 1
# train first with 2
def get_best_hid():
    for hid in np.arange(50, 120, 10):
        print(f"hidden is --- {hid}")

        #get targets
        targets = reset()
        trgt_lags = targets
        day_back(trgt_lags)

        #get target lags data set
        trgt_lags, FECOLS = create_lags_dataset(trgt_lags, LAG, OFFSET)

        display(trgt_lags)
        #get sample set
        X_train, X_test, y_train, y_test = create_train_test(trgt_lags)

        #get model results acc_per_fold, loss_per_fold , model                           hidden, dropuout_rate, learning_rate
        acc_per_fold, loss_per_fold , model = run_ARNN(X_train, X_test, y_train, y_test, hid, 0.2, 1e-2)

        #find avg loss from each fold
        loss = np.mean(loss_per_fold)
        #append to arrays
        x_hid.append(hid)
        y_hid.append(round(loss, 3))

        #delete all data sets to reset
        del trgt_lags, X_train, X_test, y_train, y_test, acc_per_fold, loss_per_fold , model

#get minmum

get_best_hid()
index = y_hid.index(min(y_hid))
best_hid = x_hid[index]
print(best_hid)

"""#### Testing different of learning rates """

x_lr = []
y_lr = []

LAG = 5
OFFSET = 1
# train first with 2
def get_best_lr():
    for lr in np.arange(0.001, 0.11, 0.02):
        print(f"lr is --- {lr}")

        #get targets
        targets = reset()
        trgt_lags = targets
        day_back(trgt_lags)

        #get target lags data set
        trgt_lags, FECOLS = create_lags_dataset(trgt_lags, LAG, OFFSET)

        display(trgt_lags)
        #get sample set
        X_train, X_test, y_train, y_test = create_train_test(trgt_lags)

        #get model results acc_per_fold, loss_per_fold , model                           hidden, dropuout_rate, learning_rate
        acc_per_fold, loss_per_fold , model = run_ARNN(X_train, X_test, y_train, y_test, 50, 0.2, lr)

        #find avg loss from each fold
        loss = np.mean(loss_per_fold)
        #append to arrays
        x_lr.append(lr)
        y_lr.append(round(loss, 3))

        #delete all data sets to reset
        del trgt_lags, X_train, X_test, y_train, y_test, acc_per_fold, loss_per_fold , model



get_best_lr()
index = y_lr.index(min(y_lr))
best_lr = x_lr[index]
print(best_lr)



"""### Auto Regressive nueral network 20 lags """

# from the best hyper parameters above we re train the models , with those parameters

#building the data again but with 20 lags
targets = reset()
# collecting what we need
trgt_lags = targets
day_back(trgt_lags)

display(trgt_lags)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #building our feature set ...
# lags = 20
# offset = 45
# trgt_lags, FECOLS = create_lags_dataset(trgt_lags , lags, offset)
# trgt_lags = trgt_lags.sort_values(by=["playerId", "engagementMetricsDate"])
# print('train sorted---')
# display(trgt_lags)
# trgt_lags = trgt_lags.dropna()
# print(trgt_lags.shape)
# gc.collect()
#

display(trgt_lags, trgt_lags.columns)

trgt_lags.fillna(-1,inplace = True)

sample_y = trgt_lags[['t1_diff','t2_diff', 't3_diff', 't4_diff']]
sample_y = sample_y.reset_index(drop=True)
sample_X = trgt_lags[FECOLS]
sample_X = sample_X.reset_index(drop=True)

display(sample_X , sample_X.columns)
display(sample_y)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2, shuffle=False)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
display(X_train , y_train)

display(X_train.shape, y_train.shape)

# train nueral network

# if hyper tuning was run , use the optimised paramters found
acc_per_fold, loss_per_fold , test_results = run_ARNN(X_train, X_test, y_train, y_test, 50, 0.2, 1e-2)

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#predictions = test_results.predict(X_test)
#display(predictions)
scores = test_results.evaluate(X_test, y_test, verbose=0)
print(f'Score for test {test_results.metrics_names[0]} of {scores[0]}')









"""### NN with  5 lags and player box scores merged """

#run each cell in order

#building the data again but with 20 lags
targets = reset()
# collecting what we need
trgt_lags = targets
day_back(trgt_lags)

#trgt_lags.reset_index()
trgt_lags

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #building our feature set ...
# lags = 5
# offset = 1
#
# trgt_lags, FECOLS = create_lags_dataset(trgt_lags , lags, offset)
# trgt_lags = trgt_lags.sort_values(by=["playerId", "engagementMetricsDate"])
# print('train sorted---')
# display(trgt_lags)
# trgt_lags = trgt_lags.dropna()
# print(trgt_lags.shape)
# gc.collect()

pl_eng_w_scores.fillna(-1,inplace = True)
display(pl_eng_w_scores)

lag_with_scores = pd.merge(trgt_lags,pl_eng_w_scores,on=['playerId','date','target1', 'target2', 'target3', 'target4' ],how = 'left').reset_index()

display(lag_with_scores)
display(lag_with_scores.columns.tolist())

##### notice how the number of rows is now 184759 from 2.5mill+ because were only taking those rows are that in
##### playerbox info

lag_with_scores = lag_with_scores.dropna()
print(lag_with_scores.shape)

lag_with_scores.fillna(-1,inplace = True)

sample_y = lag_with_scores[['t1_diff','t2_diff', 't3_diff', 't4_diff']]
sample_y = sample_y.reset_index(drop=True)
extra =["hits",'runsScored', 'homeRuns','gamesStartedPitching', 'strikeOuts','stolenBases','homeRunsPitching',"strikeOutsPitching"]

sample_X = lag_with_scores[FECOLS + extra]
sample_X = sample_X.reset_index(drop=True)

#display(sample_X , sample_X.columns)
#display(sample_y)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(sample_X, sample_y, test_size=0.2, shuffle=False)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
display(X_train , y_train)

# train nueral network
acc_per_fold, loss_per_fold , test_results = run_ARNN(X_train, X_test, y_train, y_test, 50, 0.2, 1e-2)

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

#predictions = test_results.predict(X_test)
#display(predictions)
scores = test_results.evaluate(X_test, y_test, verbose=0)
print(f'Score for test {test_results.metrics_names[0]} of {scores[0]}')

"""### Plots for data Exploration

#### Search for optimal time lag
"""

#add mean useful for analysis later
targets = pd.read_pickle("/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl")

targets['targetAvg'] = np.mean(
targets[['target1', 'target2', 'target3', 'target4']],
axis = 1)
targets["engagementMetricsDate"] = pd.to_datetime(targets["engagementMetricsDate"])

# cell 58 & 59 are courtsey of https://www.kaggle.com/arpitsolanki14/mlb-digital-engagement-data-deep-dive/notebook

autocorrel_list=list()
for i in range(25):
    ser=targets.groupby('playerId')['targetAvg'].apply(lambda x: x.autocorr(lag=i))
    df=ser.to_frame().reset_index()
    df['lag']=i
    autocorrel_list.append(df)
auto_frame=pd.concat(autocorrel_list).reset_index(drop=True)

lag_gp=auto_frame.groupby('lag').agg({'targetAvg':['mean','median']}).reset_index()
lag_gp.columns=['lag','mean','median']
fig=go.Figure()
fig.add_trace(go.Scatter(x=lag_gp['lag'],y=lag_gp['mean'],mode='lines',name='mean'))
fig.add_trace(go.Scatter(x=lag_gp['lag'],y=lag_gp['median'],mode='lines',name='median'))
fig.update_layout(title='Mean & Median of player Autocorrelation distributions across various lag periods', title_x=0.5 ,xaxis_title='lag')

"""#### Exploring player position and target average relationship """

players = pd.read_csv('/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/players.csv')

display(players)

players_info=players[['playerId','playerName','primaryPositionName']]

targets = pd.read_pickle("/Users/daksh/Desktop/sem2-2021/COMP9417-2021/project/mlb-player-digital-engagement-forecasting/nextDayPlayerEngagement_train.pkl")
targets['targetAvg'] = np.mean(
targets[['target1', 'target2', 'target3', 'target4']],
axis = 1)
targets["engagementMetricsDate"] = pd.to_datetime(targets["engagementMetricsDate"])

plys_trtg = pd.merge(targets,players,on=['playerId'],how = 'left')

display(plys_trtg)

player_high=plys_trtg.groupby('playerId').agg({'targetAvg':'mean'}).reset_index()
player_high.sort_values(by='targetAvg',inplace=True,ascending=False)

player_high3 = player_high.head(3)

top3_players=plys_trtg.loc[plys_trtg['playerId'].isin(player_high3.playerId)]

player_high100=player_high.head(100)

top100_players=pd.merge(left=player_high100,right=players_info,left_on='playerId',right_on='playerId',how='inner')
top100_players

fig=px.scatter(top100_players,y='targetAvg',color='primaryPositionName',title='TargetAvg scores for top 100 players')
fig.update_layout(title_x=0.5)
fig.show()

fig1 = px.pie(top100_players, values='targetAvg', names='primaryPositionName', title='Breakdown of top 100 players by position')
fig1.update_layout(title_x=0.5)
fig1.show()

position_avg=plys_trtg.groupby(['primaryPositionName']).agg({'targetAvg':'mean','playerId':'nunique'}).reset_index()
position_avg.sort_values(by='targetAvg',inplace=True,ascending=False)

fig=px.bar(position_avg,x='primaryPositionName',y='targetAvg',title='Target Avg values by Position')
fig.update_layout(title_x=0.5)
fig.show()
position_avg.sort_values(by='playerId',inplace=True,ascending=False)

fig1=px.bar(position_avg,x='primaryPositionName',y='playerId',title='Count of Players by Position')
fig1.update_layout(title_x=0.5)
fig1.show()


player_high_join=pd.merge(left=player_high,right=players,how='inner')
#player_high_join=player_high_join.loc[player_high_join['primaryPositionName']=='Pitcher']
player_high_join

fig2 = px.violin(player_high_join,y='targetAvg',box=True,x='primaryPositionName',title='Distribution of targetAvg by primaryPosition')
fig2.update_layout(title_x=0.5)
fig2.show()

# merge target vaerage on to player box scores group by player
extra =["playerId","targetAvg", "hits",'runsScored', 'homeRuns','gamesStartedPitching', 'strikeOuts','stolenBases','homeRunsPitching',"strikeOutsPitching"]
Box_features=pl_eng_w_scores[extra]
Box_features = Box_features.dropna()

Box_features=Box_features.groupby('playerId', '').agg({'targetAvg':'mean'}).reset_index()
player_high.sort_values(by='targetAvg',inplace=True,ascending=False)

# box_scores_corr = box_scores.corr()
# fig = px.imshow(box_scores_corr)
# fig.show()

pl_eng_w_scores.columns

run_stats=pl_eng_w_scores.groupby(['runsScored','homeRuns']).agg({'targetAvg':'mean','target1':'count'}).reset_index()
run_stats['homeRuns']=run_stats['homeRuns'].apply(str)
run_stats.columns=['runsScored','homeRuns','targetAvg','Innings']


#team_run_stats

fig = px.bar(run_stats, x="runsScored", y="targetAvg",
             color='homeRuns', barmode='group',
             height=400)
fig.update_layout(title='Relationship between targetAvg with Player runsScored & homeRuns scored in Match')
fig.update_layout(title_x=0.5)
fig.show()