import streamlit as st

# Masukkan kode yang dikonversi dari notebook di sini
def main():
    # Kode yang dihasilkan dari notebook akan dimasukkan di sini
    pass

if __name__ == '__main__':
    main()

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import re
import string
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from wordcloud import WordCloud
import os
import subprocess

# Tentukan bahwa hanya CPU yang akan digunakan
tf.config.set_visible_devices([], 'GPU')

# Perintah untuk mendownload file besar dari GitHub LFS
if not os.path.exists("training.1600000.processed.noemoticon.csv"):
    subprocess.run(["git", "lfs", "pull"])

# Download stopwords
nltk.download('stopwords')

# Function to load and process data
def load_data():
    data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")
    data.columns = ["label", "time", "date", "query", "username", "text"]
    return data

def main():
    st.title('Sentiment Analysis')

    # Load data
    data = load_data()

    # Display data
    st.write(data.head())

    # Data info
    st.write(data.columns)
    st.write(data.shape)
    st.write(data.info())

    # Check for null values
    st.write(f"Number of rows with missing values: {np.sum(data.isnull().any(axis=1))}")

    # Sentiment distribution
    sentiment_counts = data['label'].value_counts()
    st.bar_chart(sentiment_counts)

    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=['Negative', 'Positive'], autopct='%1.1f%%', startangle=140)
    st.pyplot(fig)

    # Wordcloud
    all_text = ' '.join(data['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Use a subset of data for training
    st.write("Using a subset of data for training...")
    subset_size = 10000  # Define the subset size
    data_subset = data.sample(n=subset_size, random_state=42)
    st.write(f"Subset size for training: {len(data_subset)}")
    
    # Model training
    st.write("Training the model...")
    max_words = 1000
    max_len = 150

    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(data_subset['text'].values)
    X = tokenizer.texts_to_sequences(data_subset['text'].values)
    X = sequence.pad_sequences(X, maxlen=max_len)
    Y = pd.get_dummies(data_subset['label']).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")

    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(2, name='out_layer')(layer)  # Ubah jumlah neuron menjadi 2
    layer = Activation('softmax')(layer)  # Gunakan softmax untuk multi-kelas
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=80, epochs=20, validation_split=0.1)
    st.write("Training complete!")

    # Save model
    model.save('sentiment_analysis_model.keras')  # Save model in .keras format

    # Evaluate model
    accr1 = model.evaluate(X_test, Y_test)
    st.write(f"Test set Accuracy: {accr1[1]:0.2f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Confusion Matrix
    CR = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    st.write("Confusion Matrix:")
    st.write(CR)

    fig, ax = plot_confusion_matrix(conf_mat=CR, figsize=(10, 10), show_absolute=True, show_normed=True, colorbar=True)
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC CURVE')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Metrics
    accuracy = accuracy_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    precision = precision_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    recall = recall_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    f1 = f1_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")

    # Allow user to input text for prediction
    user_input = st.text_area("Enter text for sentiment prediction")
    if st.button("Predict Sentiment"):
        if user_input:
            user_input_seq = tokenizer.texts_to_sequences([user_input])
            user_input_pad = sequence.pad_sequences(user_input_seq, maxlen=max_len)
            prediction = model.predict(user_input_pad)
            sentiment = 'Positive' if prediction.argmax(axis=1)[0] == 1 else 'Negative'
            st.write(f"Predicted Sentiment: {sentiment}")

if __name__ == '__main__':
    main()
