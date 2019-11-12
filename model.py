import pandas as pd
from collections import Counter as Counter
from nltk import WordNetLemmatizer
import re


def read_file():
    df = pd.read_csv("hn2018_2019.csv")
    df['date'] = pd.to_datetime(df['Created At'])
    start_date = '2018-01-01 00:00:00'
    end_date = '2018-12-31 00:00:00'
    mask_2018 = (df['date'] > start_date) & (df['date'] <= end_date)
    start_date = '2019-01-01 00:00:00'
    end_date = '2019-12-31 00:00:00'
    mask_2019 = (df['date'] > start_date) & (df['date'] <= end_date)
    print(df.loc[mask_2018].head(5))
    # training_data = open('2018.txt', 'a')
    # training_data.write(df.to_string())
    # training_data.close()
    #
    # training_data = open('2019.txt', 'a')
    # training_data.write(df.loc[mask_2019].to_string())
    # training_data.close()
    df = df.loc[mask_2018]
    build_vocabulary(df)
    # print(df.tail(5))


def build_vocabulary(df):
    freq_set = []

    lemmatizer = WordNetLemmatizer()

    for title in df['Title']:
        # title = re.sub('[^A-Za-z0-9]+', '', title)
        # title = re.sub(r"[^a-zA-Z0-9]+", ' ', title)
        words = title.lower().split(' ')
        for each_word in words:
            freq_set.append(lemmatizer.lemmatize(each_word))

    counts = Counter(freq_set)
    with open('frequency2.txt', 'w') as f:
        for k, v in counts.items(): f.write(f'{k} {v}\n')


if __name__ == '__main__':
    read_file()
