import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# from textacy import preprocessing
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, GRU, BatchNormalization, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#nltk.download('punkt')
#nltk.download('stopwords')
import re 
import warnings
warnings.filterwarnings('ignore')

filepath = "text.csv"
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'text': 'Text', 'label': 'Label'}, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    label_map = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
    df['Label'] = df['Label'].map(label_map)
    
    df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)\
                            .str.replace(r'[^\w\s]', '', regex=True)\
                            .str.replace(r'\s+', ' ', regex=True)\
                            .str.replace(r'\d+', '', regex=True)\
                            .str.lower()
                            
    stop = stopwords.words('english')
    df["Text"] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    return df

df = load_and_preprocess_data(filepath)


def tokenize_and_pad_texts(X_train, X_test, num_words=50000, padding_type='post'):

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)
    
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    
    maxlen = max(max(len(tokens) for tokens in X_train_sequences), max(len(tokens) for tokens in X_test_sequences))
    print("Maximum sequence length (maxlen):", maxlen)
    
    X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding=padding_type)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding=padding_type)

    return X_train_padded, X_test_padded, maxlen

def model_creat(input_Size, maxlen):

    model = Sequential()

    model.add(Embedding(input_dim=input_Size, output_dim=50, input_length=maxlen))

    model.add(Dropout(0.5))

    model.add(Bidirectional(GRU(120, return_sequences=True)))
    model.add(Bidirectional(GRU(64, return_sequences=True)))

    model.add(BatchNormalization())

    model.add(Bidirectional(GRU(64)))

    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_test_padded, y_test):
    model.evaluate(X_test_padded, y_test)
    
    y_pred = model.predict(X_test_padded)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')


if __name__ == "__main__":
    # Spilt train and test datset
    encoder = LabelEncoder()
    X = df['Text']
    y = encoder.fit_transform(df['Label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_padded, X_test_padded, maxlen = tokenize_and_pad_texts(X_train, X_test)
    input_Size = np.max(X_train_padded) + 1

    model = model_creat(input_Size, maxlen)
    history = model.fit(X_train_padded, y_train, epochs=5, 
                        batch_size=1500, validation_data=(X_test_padded, y_test))
    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy'])) + 1

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    # Plot training and validation accuracy
    axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axs[0].scatter(best_epoch - 1, history.history['val_accuracy'][best_epoch - 1], color='green', label=f'Best Epoch: {best_epoch}')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend()
    # Plot training and validation loss
    axs[1].plot(history.history['loss'], label='Training Loss', color='blue')
    axs[1].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axs[1].scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='green',label=f'Best Epoch: {best_epoch}')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    evaluate_model(model, X_test_padded, y_test)

