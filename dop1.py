
import os
import random
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.config.experimental.enable_op_determinism()

def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first





class BinaryClassfier(Model):
  def __init__(self, hp):
    super(BinaryClassfier, self).__init__()

    self.encoder = tf.keras.Sequential([*[layers.Dense(i, activation='relu' ) for i in hp], layers.Dense(1, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    return encoded

  
def model_builder(hp):
  model = BinaryClassfier(hp)
  model.compile(optimizer='adam', loss='mse', metrics= ["accuracy"])

  return model

class ProgressCallback(Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.epoch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        self.progress_bar.progress(int(self.epoch_count / self.epochs * 100))

def predict(model, data, threshold):
  reconstructions = model.predict(data)
  return [not i  for i in tf.math.less(reconstructions, threshold)]
        
# prepare input data
def one_hot_encode_dataframe(df):
  """
  Automatically one-hot encodes all categorical columns in a Pandas DataFrame.

  Args:
    df: The Pandas DataFrame.

  Returns:
    A new Pandas DataFrame with one-hot encoded columns.
  """

  categorical_cols = df.select_dtypes(include=['object', 'category']).columns
  df_encoded = df.copy()

  for col in categorical_cols:
    unique_values = df[col].unique()
    mapping = {value: index for index, value in enumerate(unique_values)}
    df_encoded[col] = df_encoded[col].map(mapping)
    encoded_col = tf.keras.utils.to_categorical(df_encoded[col], num_classes=len(unique_values))
    encoded_df = pd.DataFrame(encoded_col, columns=[f"{col}_{value}" for value in unique_values])
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1).drop(col, axis=1)
  return df_encoded


st.title('Визуальная работа с ML алгоритмами')

uploaded_file = st.file_uploader("ВЫберити файл для работы с ")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    try:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
        columns = st.multiselect("Выберите колонки:",dataframe.columns)
        filter = st.radio("Способ выбора:", ("включение","исключение"))

        if filter == "исключение":
            columns = [col for col in dataframe.columns if col not in columns]

        dataframe = dataframe[columns]
        dataframe = one_hot_encode_dataframe(dataframe)
        
        st.write("Итоговый датасет:", dataframe)
        
        if (len(columns) > 0):
            columns_selected = st.selectbox("Выберите целевую колонку:",dataframe.columns)
            num = st.slider("Выберите количество скрытх слоев", 1, 5, 1)
            hp = []
            for i in range(num):
                s = st.slider("Выберите размер скрытого слоя " + str(i + 1), 1, len(columns)- 1, (len(columns) -1) // 2, key=i)
                hp.append(s)
            split = st.slider("Выберите размер тестовой выборки", 0.0, 1.0, .2)
            epochs = st.slider("Выберите количество эпох", 0, 200, 10)
            
            if st.button("Начать"):
                setup_seed(1)
                autoencoder = model_builder(hp)
                without = dataframe.drop(columns=[columns_selected])
                target = dataframe[columns_selected]
                
                
                train_data, test_data, train_labels, test_labels = train_test_split(
                        without, target, test_size=split, random_state=1
                )
                progress_callback = ProgressCallback(epochs)
                print(train_data, train_labels)
                history = autoencoder.fit(train_data, train_labels, epochs=epochs,
                validation_data=(test_data, test_labels), callbacks=[progress_callback])
                fig, ax = plt.subplots()
                plt.plot(history.history["loss"], label="Training Loss")
                plt.plot(history.history["val_loss"], label="Validation Loss")
                plt.legend()

                st.pyplot(fig)
                
                from sklearn.metrics import ConfusionMatrixDisplay
                preds = predict(autoencoder, test_data, 0.5)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(test_labels, preds).plot(ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Выберите хотя бы 1 стобец")
        
    except Exception as e:
        st.write(e)