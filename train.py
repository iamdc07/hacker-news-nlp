import pandas as pd
from collections import Counter, OrderedDict
import nltk
import time


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

    # for title in df['Title'].head(50):
    #     training_data = open('sample.txt', 'a')
    #     training_data.write(title + "\n")
    #     training_data.close()

    # training_data = open('2018.txt', 'a')
    # training_data.write(df.to_string())
    # training_data.close()
    #
    # training_data = open('2019.txt', 'a')
    # training_data.write(df.loc[mask_2019].to_string())
    # training_data.close()
    df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]

    build_vocabulary(pd.read_csv('sample.csv'))
    # print(df.tail(5))


def build_vocabulary(df):
    # word_list = []

    lemmatizer = nltk.WordNetLemmatizer()

    word_freq_dict = {}

    j = int(0)

    start_time = time.process_time()

    for title in df['Title']:
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)
        # tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*")
        # tokenizer = nltk.RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
        # raw = nltk.word_tokenize(title.lower())

        raw = tokenizer.tokenize(title.lower())
        # raw = nltk.TreebankWordTokenizer().tokenize(title.lower())
        # raw = nltk.WhitespaceTokenizer().tokenize(title.lower())

        bigrm = list(nltk.bigrams(title.split()))
        pos = nltk.pos_tag(raw)
        pos_dict = dict(pos)
        bigrams = []
        for i in bigrm:
            bigrams.append((''.join([w + ' ' for w in i])).strip())

        # print('Bigrm:', bigrams)
        # print('Pos:', pos)
        # print('Pos1:', pos_dict)

        for each_element in bigrams:
            word = each_element.split(' ')
            # print("HERE:", word)
            if word[0].istitle() and word[1].istitle():
                if word[0].lower() in raw and (
                        pos_dict.get(word[0].lower()) == 'NN' or pos_dict.get(word[0].lower()) == 'NNS'):
                    if word[1].lower() in raw and (
                            pos_dict.get(word[1].lower()) == 'NN' or pos_dict.get(word[1].lower()) == 'NNS'):
                        index1 = raw.index(word[0].lower())
                        # print('WORD0:', word[0])
                        # print('WORD1:', word[1])
                        # print('INDEX1:', index1)
                        del raw[index1]
                        if word[1].lower() in raw:
                            index2 = raw.index(word[1].lower())
                        else:
                            raw.append(word[0])
                            continue
                        # print('INDEX2:', index2)
                        # print('RAW:', raw)
                        del raw[index2]
                        # print('RAW:', raw)
                        temp = each_element.lower() + "-" + df.at[j, 'Post Type']
                        freq = word_freq_dict.get(temp)

                        if freq is None:
                            word_freq_dict[temp] = 1
                        else:
                            freq += 1
                            word_freq_dict[temp] = freq

                        # word_list.append(each_element.lower())

        pos = nltk.pos_tag(raw)

        # print("TAG:", nltk.pos_tag(raw))

        for each_word in pos:
            wordnet_tag = get_wordnet_pos(each_word[1])
            # print("WORD:", each_word[0])
            # print("TAG:", each_word[1])
            word_lemm = lemmatizer.lemmatize(each_word[0], wordnet_tag)
            temp = word_lemm + "-" + df.at[j, 'Post Type']
            value = word_freq_dict.get(temp)

            if value is None:
                word_freq_dict[temp] = 1
            else:
                value += 1
                word_freq_dict[temp] = value

            # word_list.append(lemmatizer.lemmatize(each_word[0], wordnet_tag))
            # break

        # print("SENTENCE:", title)
        # print("TOKENIZED:", word_list)
        # input('PRESS A KEY!')

        j += 1
        pos.clear()

    # word_list.sort()
    # counts = Counter(word_list)

    od = OrderedDict(sorted(word_freq_dict.items()))

    total_time = time.process_time() - start_time

    print('TOTAL TIME TAKEN IN (S):', total_time)
    # a = dict(sorted(counts.items()))

    with open('frequency_dict.txt', 'w') as file:
        for key, val in od.items():
            file.write(str(key) + " " + str(val) + "\n")

    # with open('frequency_ngram.txt', 'w') as f:
    #     for k, v in counts.items(): f.write(f'{k} {v}\n')

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
