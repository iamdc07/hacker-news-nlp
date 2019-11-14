import pandas as pd
from collections import Counter as Counter
import nltk


def read_file():
    df = pd.read_csv("hn2018_2019.csv")
    df['date'] = pd.to_datetime(df['Created At'])
    start_date = '2018-01-01 00:00:00'
    end_date = '2018-12-31 00:00:00'
    mask_2018 = (df['date'] > start_date) & (df['date'] <= end_date)
    start_date = '2019-01-01 00:00:00'
    end_date = '2019-12-31 00:00:00'
    mask_2019 = (df['date'] > start_date) & (df['date'] <= end_date)
    # print(df.loc[mask_2018].head(5))

    for title in df['Title'].head(50):
        training_data = open('sample.txt', 'a')
        training_data.write(title + "\n")
        training_data.close()


    # training_data = open('2018.txt', 'a')
    # training_data.write(df.to_string())
    # training_data.close()
    #
    # training_data = open('2019.txt', 'a')
    # training_data.write(df.loc[mask_2019].to_string())
    # training_data.close()
    df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]
    build_vocabulary(df_training)
    # print(df.tail(5))


def build_vocabulary(df):
    word_list = []

    lemmatizer = nltk.WordNetLemmatizer()

    for title in df['Title']:
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)
        # tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*")
        # tokenizer = nltk.RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
        # raw = nltk.word_tokenize(title.lower())

        raw = tokenizer.tokenize(title.lower())
        # raw = nltk.TreebankWordTokenizer().tokenize(title.lower())
        # raw = nltk.WhitespaceTokenizer().tokenize(title.lower())

        # print("SPECIAL CHARACTERS:", special_char)

        pos = nltk.pos_tag(raw)

        # print("TAG:", nltk.pos_tag(raw))

        for each_word in pos:
            wordnet_tag = get_wordnet_pos(each_word[1])
            # print("WORD:", each_word[0])
            # print("TAG:", each_word[1])
            word_list.append(lemmatizer.lemmatize(each_word[0], wordnet_tag))

        # print("SENTENCE:", title)
        # print("TOKENIZED:", word_list)
        # input('PRESS A KEY!')

        pos.clear()

    word_list.sort()
    counts = Counter(word_list)
    # a = dict(sorted(counts.items()))

    # print(word_list.count())
    with open('frequency_new.txt', 'w') as f:
        for k, v in counts.items(): f.write(f'{k} {v}\n')

    print("TITLE:", df.at[0, 'Title'])


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return nltk.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return nltk.wordnet.NOUN


if __name__ == '__main__':
    read_file()
