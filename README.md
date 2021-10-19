# MLB Player Digital Engagement Forecasting

Kaggle Competition:

[Link to Competition on Kaggle](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting)

[Link to google drive](https://drive.google.com/file/d/1F5zvl8ftFPtUerAlJzbabGjJcmFoomrv/view?usp=sharing)

The google drive contains all relevant files and data required for the project 

## Goal: 

*Predict next day Fan Engagement with a baseball player's digital content i.e. Social Media Engagement for players*

### High-Level Methodology

The following steps are implemented in this notebook:

Data Preparation

- Split the whole dataset into 80% training and 20% testing with timeseries split
- remove erroneous values  

Model Training
- Apply 10-fold stratified cross-validation to avoid overfitting
- Apply either Grid Search or Random Search to fine-tune hyperparameters

Model Evaluation
- Select the final model based on mean column-wise mean absolute error (MCMAE).


  
