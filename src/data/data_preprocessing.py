import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
import emoji
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df, col='caption'):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = ''.join([char for char in text if not char.isdigit()])
        text = text.lower()
        text = emoji.replace_emoji(text,replace='')
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        text = " ".join([word for word in text.split() if word not in stop_words])
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    df[col] = df[col].apply(preprocess_text)

    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df

def main():
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        train_processed_data = preprocess_dataframe(train_data, 'content')
        test_processed_data = preprocess_dataframe(test_data, 'content')

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()