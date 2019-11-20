import pandas as pd
import nltk, math
import experiments
import time
import operator
from collections import OrderedDict

remove_freq = 1
remove_percent = 0
smoothing_value = 0


# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


def read_file(exp=1):
    global df_testing
    global df_training

    df = pd.read_csv("./hn2018_2019.csv")

    df['date'] = pd.to_datetime(df['Created At'])
    start_date = '2018-01-01 00:00:00'
    end_date = '2018-12-31 00:00:00'
    mask_2018 = (df['date'] > start_date) & (df['date'] <= end_date)
    start_date = '2019-01-01 00:00:00'
    end_date = '2019-12-31 00:00:00'
    mask_2019 = (df['date'] > start_date) & (df['date'] <= end_date)
    # df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]
    df_training = pd.read_csv("./sample.csv")

    build_vocabulary(df_training, exp)


def build_vocabulary(df, exp):
    global word_freq_dict
    global start_time
    word_freq_dict = {}

    j = 0

    start_time = time.process_time()

    if exp == 2:
        stop_words_df = pd.read_csv("./Stopwords.txt")
        stop_words = stop_words_df["a"].tolist()

    for title in df['Title']:
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(title.lower())

        if exp == 2:
            raw = list(set(raw).difference(stop_words))
            title = ' '.join([str(elem) for elem in raw])
        elif exp == 3:
            for each in raw:
                if len(each) >= 9 or len(each) <= 2:
                    raw.remove(each)

        j = tokenize_word(raw, title, df, j)

    od = OrderedDict(sorted(word_freq_dict.items()))

    with open('frequency_dict.txt', 'w') as file:
        for key, val in od.items():
            file.write(str(key) + " " + str(val) + "\n")

    train(od, exp)

    total_time = time.process_time() - start_time
    print('TOTAL TIME TAKEN IN (S):', total_time)
    print('TOTAL TIME TAKEN IN (MINUTES):', total_time / 60)


def tokenize_word(raw, title, df, j, testing=False):
    words_removed = []
    bigrams = []
    word_list = []

    lemmatizer = nltk.WordNetLemmatizer()

    bigrm = list(nltk.bigrams(title.split()))
    pos = nltk.pos_tag(raw)
    pos_dict = dict(pos)

    for i in bigrm:
        bigrams.append((''.join([w + ' ' for w in i])).strip())

    for each_element in bigrams:
        word = each_element.split(' ')

        indices_0 = [i for i, x in enumerate(raw) if x == word[0]]
        indices_1 = [i for i, x in enumerate(raw) if x == word[1]]

        if word[0].istitle() and word[1].istitle():
            if len(indices_0) > 0 and (
                    pos_dict.get(word[0].lower()) == 'NN' or pos_dict.get(word[0].lower()) == 'NNS'):
                if len(indices_1) > 0 and (
                        pos_dict.get(word[1].lower()) == 'NN' or pos_dict.get(word[1].lower()) == 'NNS'):
                    raw.remove(word[0].lower())
                    raw.index(word[1].lower())
                    raw.remove(word[1].lower())

                    if testing is False:
                        temp = each_element.lower() + "-" + df.at[j, 'Post Type']
                        freq = word_freq_dict.get(temp)
                        if freq is None:
                            word_freq_dict[temp] = 1
                        else:
                            freq += 1
                            word_freq_dict[temp] = freq
                    else:
                        word_list.append(each_element.lower())

    pos = nltk.pos_tag(raw)

    for each_word in pos:
        wordnet_tag = get_wordnet_pos(each_word[1])

        if each_word[1] == "FW" or each_word[1] == "CD":
            words_removed.append(each_word[0])
            continue
        if len(each_word[0]) == 1 and not (each_word[0] == "a" or each_word[0] == "i"):
            words_removed.append(each_word[0])
            continue

        word_lemm = lemmatizer.lemmatize(each_word[0], wordnet_tag)

        if testing is False:
            temp = word_lemm + "-" + df.at[j, 'Post Type']
            value = word_freq_dict.get(temp)
            if value is None:
                word_freq_dict[temp] = 1
            else:
                value += 1
                word_freq_dict[temp] = value
        else:
            if testing:
                word_list.append(word_lemm)

    j += 1
    pos.clear()

    with open("./vocabulary.txt", "w") as file:
        for element in words_removed:
            file.write(element)

    if testing:
        return j, word_list
    else:
        return j


def train(freq_dict, exp):
    word = []
    word_list = []
    post_type = []
    p_ask_hn_dict = {}
    p_story_dict = {}
    p_show_hn_dict = {}
    p_poll_dict = {}
    class_probability = []
    smoothing = 0.5

    if exp == 4:
        new_dict = {k: v for k, v in freq_dict.items() if not (v <= remove_freq)}
        freq_dict = new_dict
    elif exp == 4.5:
        sorted_dict_list = sorted(freq_dict.items(), key=operator.itemgetter(1))
        remove_elements = int(len(sorted_dict_list) * remove_percent)
        new_dict_list = sorted_dict_list[remove_elements:]
        # print("OLD LIST SIZE:", len(sorted_dict_list), " NEW LIST SIZE:", len(new_dict_list), " REMOVE:",
        #       remove_elements)
        freq_dict = dict(new_dict_list)
        # print(freq_dict)
        # exit(0)
    elif exp == 5:
        smoothing = smoothing_value

    dict_keys = freq_dict.keys()
    freq = list(freq_dict.values())

    for each in dict_keys:
        word_class = each.split('-')
        word.append(word_class[0])
        post_type.append(word_class[1])

    df = pd.DataFrame({'Word': word, 'Class': post_type, 'Frequency': freq})
    # df.to_csv("vocabulary.csv")
    story_df = df[df.Class.str.match('story', case=False)]
    ask_hn_df = df[df.Class.str.match('ask_hn', case=False)]
    show_hn_df = df[df.Class.str.match('show_hn', case=False)]
    poll_df = df[df.Class.str.match('poll', case=False)]

    story_dft = df_training[df_training["Post Type"].str.match('story', case=False)]
    ask_hn_dft = df_training[df_training["Post Type"].str.match('ask_hn', case=False)]
    show_hn_dft = df_training[df_training["Post Type"].str.match('show_hn', case=False)]
    poll_dft = df_training[df_training["Post Type"].str.match('poll', case=False)]

    show_hn_words = dict(zip(show_hn_df.Word, show_hn_df.Frequency))
    ask_hn_words = dict(zip(ask_hn_df.Word, ask_hn_df.Frequency))
    poll_words = dict(zip(poll_df.Word, poll_df.Frequency))
    story_words = dict(zip(story_df.Word, story_df.Frequency))

    show_hn_count = sum(show_hn_words.values())
    ask_hn_count = sum(ask_hn_words.values())
    poll_count = sum(poll_words.values())
    story_count = sum(story_words.values())

    total_words = df.Frequency.sum()
    vocabulary = df.Word.unique()
    vocabulary_size = len(vocabulary)
    experiments.no_of_words = vocabulary_size

    # int(15.55555 * 10 ** 3) / 10.0 ** 3

    print("Size:", vocabulary_size)

    story_dft.to_csv("./story_dft.csv")
    ask_hn_dft.to_csv("./ask_hn_dft.csv")
    show_hn_dft.to_csv("./show_hn_dft.csv")
    poll_dft.to_csv("./poll_dft.csv")

    class_probability_show_hn = (len(show_hn_dft.index) / len(df_training.index))
    class_probability_ask_hn = (len(ask_hn_dft.index) / len(df_training.index))
    class_probability_poll = (len(poll_dft.index) / len(df_training.index))
    class_probability_story = (len(story_dft.index) / len(df_training.index))

    # print("class_probability_show_hn: ", class_probability_show_hn)
    # print("class_probability_ask_hn: ", class_probability_ask_hn)
    # print("class_probability_poll: ", class_probability_poll)
    # print("class_probability_story: ", class_probability_story)

    # class_probability_show_hn = class_probability_show_hn
    # class_probability_ask_hn = class_probability_ask_hn
    # class_probability_poll = class_probability_poll
    # class_probability_story = class_probability_story

    # class_probability_show_hn = int(class_probability_show_hn * 10 ** 10) / 10.0 ** 10
    # class_probability_ask_hn = int(class_probability_ask_hn * 10 ** 10) / 10.0 ** 10
    # class_probability_poll = int(class_probability_poll * 10 ** 10) / 10.0 ** 10
    # class_probability_story = int(class_probability_story * 10 ** 10) / 10.0 ** 10

    line_count = 1

    for word in vocabulary:
        temp_show_hn_freq = show_hn_words[word] if word in show_hn_words else 0

        temp_ask_hn_freq = ask_hn_words[word] if word in ask_hn_words else 0

        temp_story_freq = story_words[word] if word in story_words else 0

        temp_poll_freq = poll_words[word] if word in poll_words else 0

        p_word_given_show_hn = ((temp_show_hn_freq + smoothing) / (show_hn_count + vocabulary_size))

        p_word_given_ask_hn = ((temp_ask_hn_freq + smoothing) / (ask_hn_count + vocabulary_size))

        p_word_given_poll = ((temp_poll_freq + smoothing) / (poll_count + vocabulary_size))

        p_word_given_story = ((temp_story_freq + smoothing) / (story_count + vocabulary_size))

        p_word_given_show_hn = p_word_given_show_hn
        p_word_given_ask_hn = p_word_given_ask_hn
        p_word_given_poll = p_word_given_poll
        p_word_given_story = p_word_given_story

        # p_word_given_show_hn = int(p_word_given_show_hn * 10 ** 10) / 10.0 ** 10
        #
        # p_word_given_ask_hn = int(p_word_given_ask_hn * 10 ** 10) / 10.0 ** 10
        #
        # p_word_given_poll = int(p_word_given_poll * 10 ** 10) / 10.0 ** 10
        #
        # p_word_given_story = int(p_word_given_story * 10 ** 10) / 10.0 ** 10

        if exp == 1:
            file = open("model-2018.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close()
        elif exp == 2:
            file = open("stopword-model.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close()
        elif exp == 3:
            file = open("wordlength-model.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
            file.close()
        line_count += 1

        p_ask_hn_dict[word] = p_word_given_ask_hn
        p_show_hn_dict[word] = p_word_given_show_hn
        p_poll_dict[word] = p_word_given_poll
        p_story_dict[word] = p_word_given_story
        word_list.append(word)

    end_time = time.process_time() - start_time
    print("Time to train:", end_time)

    # 0: show_hn
    # 1: ask_hn
    # 2: poll
    # 3: story

    class_probability.append(class_probability_show_hn)
    class_probability.append(class_probability_ask_hn)
    class_probability.append(class_probability_poll)
    class_probability.append(class_probability_story)

    accuracy = experiments.baseline(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
                                    p_story_dict, exp)

    if exp == 4 or exp == 4.5:
        experiments.each_accuracy = accuracy


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
    experiments.select_experiment()
