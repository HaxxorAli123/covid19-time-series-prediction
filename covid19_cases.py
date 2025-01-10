# %%
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())

# %%
from helper_module import WindowGenerator

# %%
import datetime
import IPython
import pickle
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
import mlflow.tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# %%
os.getcwd()
os.chdir(r"c:\\Users\\User\\OneDrive\\Desktop\\YP\\Subjects\\Capstone\\project1")
os.getcwd()

# %%
CSV_PATH = os.path.join(os.getcwd(),'datasets','cases_malaysia_train.csv')
CSV_PATH_TEST = os.path.join(os.getcwd(),'datasets','cases_malaysia_test.csv')
cases_df = pd.read_csv(CSV_PATH)
test_df = pd.read_csv(CSV_PATH_TEST)
print(cases_df.head())

# %%
#removing the column date from cases_df
date_time = pd.to_datetime(cases_df.pop('date'), format='%d/%m/%Y')
date_time = pd.to_datetime(test_df.pop('date'), format='%d/%m/%Y')
print(date_time.dtype)
print(date_time.dtype)

# %%
cases_df.info()
cases_df.isna().sum()

# %%
print(cases_df[cases_df['cases_new']=="?"])
print(cases_df[cases_df['cases_new']==" "])

# %%
cases_df = cases_df.replace('?', np.nan)
cases_df = cases_df.replace(" ", np.nan)

print(cases_df[cases_df['cases_new'].isnull()])

# %%
row_numbers = cases_df[cases_df['cases_new'].isnull()].index
print(row_numbers)

# %%
for row_num in row_numbers:
    cases_df.loc[row_num,'cases_new'] = (cases_df.loc[row_num,'cases_active']-cases_df.loc[row_num-1,'cases_active'])+cases_df.loc[row_num,'cases_recovered']

# %%
print(cases_df.info())
cases_df['cases_new'] = cases_df['cases_new'].astype('float64')
print(cases_df.info())

# %%
print(test_df.loc[test_df['cases_new'].isnull()])
row_index = test_df.loc[test_df['cases_new'].isnull()].index
row_index = row_index[0]
print(row_index)

# %%
print(test_df.loc[row_index,'cases_new'])
test_df.loc[row_index,'cases_new'] = (test_df.loc[row_index,'cases_active']-test_df.loc[row_index-1,'cases_active'])+test_df.loc[row_index,'cases_recovered']
print(test_df.loc[row_index,'cases_new'])

# %%
print(test_df.info())

# %%
train_case_df = cases_df.astype('float64')
test_case_df = test_df.astype('float64')

# %%
print(train_case_df.info())
print(test_case_df.info())

# %%
train_case_df_copy = train_case_df.copy()
test_case_df_copy = test_case_df.copy()
train_date_time = train_case_df_copy.index  
test_date_time = test_case_df_copy.index  

# %%
train_case_df_copy.plot(subplots=True,figsize=(10,20))
plt.show()

# %%
test_case_df_copy.plot(subplots=True,figsize=(10,20))
plt.show()

# %%
# [Data Splitting] split out the dataset accordingly to split ratio to prevent the data from shuffled using 
# functions such as train_test_split
data_size = test_df.shape[0]
val_ratio = 0.5
test_ratio = 0.5

val_size = int(data_size * val_ratio)
test_size = data_size - val_size

train_df = train_case_df
val_df = test_df.iloc[:val_size]
test_df = test_df.iloc[val_size:]

print("Validation set size:",len(val_df))
print("Test set size:",len(test_df))
print("Training set size:",len(cases_df))

# %%
#Filling up Nan values in columns_to_fill with '0'
columns_to_fill = [
    'cluster_import', 'cluster_religious', 'cluster_community',
    'cluster_highRisk', 'cluster_education', 'cluster_detentionCentre', 'cluster_workplace'
]
train_df[columns_to_fill] = train_df[columns_to_fill].fillna(0)
print(train_df.info())

# %%
# [Data normalization] Perform this step using pandas way
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# %%
#This is a single-step single-output problem. With an input window width of 30, 
#output window width of 30 and an offset of 1.
single_window = WindowGenerator(input_width=30,
                                   label_width=30,
                                   shift = 1,
                                   train_df =train_df,
                                   val_df =val_df,
                                   test_df =test_df,
                                   label_columns=["cases_new"])
single_window.plot(plot_col="cases_new")
print(single_window.train.element_spec[1])

# %%
# [MLOps] Create MLFlow experiment
mlflow.set_experiment("Capstone Covid19")

# %%
model_single_32 = keras.Sequential()
model_single_32.add(keras.layers.LSTM(32,activation='relu',return_sequences=True))
model_single_32.add(keras.layers.Dense(1))
model_single_32.summary()

model_single_32.compile(optimizer='adam',loss='mse',metrics=['mae'])

# %%
model_single_64 = keras.Sequential()
model_single_64.add(keras.layers.LSTM(64,activation='relu',return_sequences=True))
model_single_64.add(keras.layers.Dense(1))
model_single_64.summary()

model_single_64.compile(optimizer='adam',loss='mse',metrics=['mae'])

# %%
model_single_128 = keras.Sequential()
model_single_128.add(keras.layers.LSTM(128,activation='relu',return_sequences=True))
model_single_128.add(keras.layers.Dense(1))
model_single_128.summary()

model_single_128.compile(optimizer='adam',loss='mse',metrics=['mae'])

# %%
# Create the single time step model
model_single_256 = keras.Sequential()
model_single_256.add(keras.layers.LSTM(256,activation='relu',return_sequences=True))
model_single_256.add(keras.layers.Dense(1))
model_single_256.summary()
model_single_256.compile(optimizer='adamw',loss='mse',metrics=['mae'])

# %%
# train the model 
MAX_EPOCH = 70

# %%
with mlflow.start_run(run_name="lstm_32") as run:
    mlflow.tensorflow.autolog()
    history_32 = model_single_32.fit(single_window.train,validation_data=single_window.val,epochs=MAX_EPOCH)

# %%
with mlflow.start_run(run_name="lstm_64") as run:
    mlflow.tensorflow.autolog()
    history_64 = model_single_64.fit(single_window.train,validation_data=single_window.val,epochs=MAX_EPOCH)

# %%
with mlflow.start_run(run_name="lstm_128") as run:
    mlflow.tensorflow.autolog()
    history_128 = model_single_128.fit(single_window.train,validation_data=single_window.val,epochs=MAX_EPOCH)

# %%
with mlflow.start_run(run_name="lstm_256") as run:
    mlflow.tensorflow.autolog()
    history_256 = model_single_256.fit(single_window.train,validation_data=single_window.val,epochs=MAX_EPOCH)

# %%
# Check training result
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(history_32.epoch,history_32.history['loss'])
plt.plot(history_32.epoch,history_32.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss graph')
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(history_64.epoch,history_64.history['loss'])
plt.plot(history_64.epoch,history_64.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss graph')
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(history_128.epoch,history_128.history['loss'])
plt.plot(history_128.epoch,history_128.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss graph')
plt.show()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.plot(history_256.epoch,history_256.history['loss'])
plt.plot(history_256.epoch,history_256.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss graph')
plt.show()

# %%
model_load = mlflow.tensorflow.load_model(model_uri=f"models:/covid19_predictor/1")
type(model_load)

# %%
predictions = model_load.predict(single_window.test)
predictions_squeezed = predictions.squeeze(axis=-1)  # Remove the last dimension
predictions_df = pd.DataFrame(predictions_squeezed)
print(predictions_df)

# %%
single_window.plot(plot_col='cases_new',model=model_load)


