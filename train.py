import pandas as pd
from collections import Counter, OrderedDict
import nltk, math
import time


def read_file():
    df = pd.read_csv("./hn2018_2019.csv")
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

    build_vocabulary(pd.read_csv('./sample.csv'))
    # print(df.tail(5))


def build_vocabulary(df):
    # word_list = []

    lemmatizer = nltk.WordNetLemmatizer()

    word_freq_dict = {}

    j = int(0)

    start_time = time.process_time()

    # freq_df = pd.DataFrame(columns=['Word', 'Post-Type', 'Frequency'])
    # print(freq_df)
    # exit(0)

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
                        # if word[1].lower() in raw:
                        index2 = raw.index(word[1].lower())
                        # else:
                        #     print("ndkcha")
                        #     raw.append(word[0])
                        #     continue
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

            # if wordnet_tag == "IGNORE":
            #     continue
            # else:
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

    # a = dict(sorted(counts.items()))

    # with open('frequency_dict.txt', 'w') as file:
    #     for key, val in od.items():
    #         file.write(str(key) + " " + str(val) + "\n")

    train(od)

    total_time = time.process_time() - start_time
    print('TOTAL TIME TAKEN IN (S):', total_time)

    # with open('frequency_ngram.txt', 'w') as f:
    #     for k, v in counts.items(): f.write(f'{k} {v}\n')

    print("TITLE:", df.at[0, 'Title'])


def train(freq_dict):
    dict_keys = freq_dict.keys()
    freq = list(freq_dict.values())
    word = []
    post_type = []

    for each in dict_keys:
        word_class = each.split('-')
        word.append(word_class[0])
        post_type.append(word_class[1])

    df = pd.DataFrame({'Word': word, 'Class': post_type, 'Frequency': freq})
    story_df = df[df.Class.str.contains('story', case=False)]
    ask_hn_df = df[df.Class.str.contains('ask_hn', case=False)]
    show_hn_df = df[df.Class.str.contains('show_hn', case=False)]
    poll_df = df[df.Class.str.contains('poll', case=False)]

    show_hn_count = show_hn_df['Frequency'].sum()
    ask_hn_count = ask_hn_df['Frequency'].sum()
    story_count = story_df['Frequency'].sum()
    poll_count = poll_df['Frequency'].sum()

    vocabulary_size = len(df.Word.unique())

    # temp_df_show_hn = show_hn_df[show_hn_df['Word'].str.contains('Domain', regex=False, case=False, na=False)]['Frequency'].tolist()
    # if len(temp_df_show_hn) == 0:
    #     temp_df_show_hn.append(0)
    # print(temp_df_show_hn[0])
    # print(type(int(temp_df_show_hn['Frequency'])))

    int(15.55555 * 10 ** 3) / 10.0 ** 3

    print("Size:", vocabulary_size)

    line_count = 1
    with open('model-2018.txt', 'w') as file:
        for key, freq in freq_dict.items():
            word = key.split('-')

            temp_df_show_hn = show_hn_df[show_hn_df['Word'].str.contains(word[0], regex=False, case=False, na=False)][
                'Frequency'].tolist()
            if len(temp_df_show_hn) == 0:
                temp_df_show_hn.append(0)
            p_word_given_show_hn = int(((temp_df_show_hn[0] + 0.5) / (show_hn_count + vocabulary_size)) * 10 ** 10) / 10.0 ** 10

            temp_df_ask_hn = ask_hn_df[ask_hn_df['Word'].str.contains(word[0], regex=False, case=False, na=False)][
                'Frequency'].tolist()
            if len(temp_df_ask_hn) == 0:
                temp_df_ask_hn.append(0)
            p_word_given_ask_hn = int(((temp_df_ask_hn[0] + 0.5) / (ask_hn_count + vocabulary_size)) * 10 ** 10) / 10.0 ** 10

            temp_df_story = story_df[story_df['Word'].str.contains(word[0], regex=False, case=False, na=False)][
                'Frequency'].tolist()
            if len(temp_df_story) == 0:
                temp_df_story.append(0)
            p_word_given_story = int(((temp_df_story[0] + 0.5) / (story_count + vocabulary_size) ) * 10 ** 10) / 10.0 ** 10

            temp_df_poll = poll_df[poll_df['Word'].str.contains(word[0], regex=False, case=False, na=False)][
                'Frequency'].tolist()
            if len(temp_df_poll) == 0:
                temp_df_poll.append(0)
            p_word_given_poll = int(((temp_df_poll[0] + 0.5) / (poll_count + vocabulary_size)) * 10 ** 10) / 10.0 ** 10

            file.write(str(line_count) + " | w " + str(word[0]) + " | f " + str(freq) + " | p " + str(
                temp_df_story[0]) + " | " + str(
                p_word_given_story) + " " + str(temp_df_ask_hn[0]) + " " + str(
                p_word_given_ask_hn) + " " + str(
                temp_df_show_hn[0]) + " " + str(p_word_given_show_hn) + " " + str(
                temp_df_poll[0]) + " " + str(
                p_word_given_poll) + " " + '\n')

            file.write(str(line_count) + " | w " + str(word[0]) + " | f " + str(freq) + " | p " + str(
                p_word_given_story) + " | " + str(
                temp_df_ask_hn[0]) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_df_show_hn[0]) + " " + str(
                p_word_given_show_hn) + " " + str(temp_df_poll[0]) + " " + str(
                p_word_given_poll) + " " + str(
                p_word_given_poll) + " " + '\n')
            line_count += 1

    # print(story_df)
    # print(df)

    # with open('probability.txt', 'w') as file:
    #     for row in df:
    #         file.write(str(row["Word"]) + str(row["Class"]) + "\n")

    # with open('unique.txt', 'w') as file:
    #     for row in vocabulary_size:
    #         file.write(str(row) + '\n')

    # story_df.to_csv('./story.csv')
    # ask_hn_df.to_csv('./ask_hn.csv')
    # show_hn_df.to_csv('./show_hn.csv')
    # poll_df.to_csv('./poll.csv')


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