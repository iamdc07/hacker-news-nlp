import pandas as pd
import nltk
import experiments
import time
import operator
from collections import OrderedDict
import warnings


warnings.filterwarnings('ignore')
remove_freq = 1
remove_percent = 0
smoothing_value = 0


# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


def read_file(exp=1):
    global df_testing
    global df_training

    df = pd.read_csv("./hn2018_2019.csv")
    # df = pd.read_csv("./sample.csv")
    df = df.drop(columns=df.columns[0])
    df['date'] = pd.to_datetime(df['Created At'])
    start_date = '2018-01-01 00:00:00'
    end_date = '2018-12-31 00:00:00'
    mask_2018 = (df['date'] > start_date) & (df['date'] <= end_date)
    start_date = '2019-01-01 00:00:00'
    end_date = '2019-12-31 00:00:00'
    mask_2019 = (df['date'] > start_date) & (df['date'] <= end_date)
    df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]

    build_vocabulary(df_training, exp)


def build_vocabulary(df, exp):
    global word_freq_dict
    global start_time
    global words_removed
    word_freq_dict = {}
    words_removed = set()

    start_time = time.process_time()

    if exp == 2:
        stop_words_df = pd.read_csv("./Stopwords.txt")
        stop_words = stop_words_df["a"].tolist()

    for index, row in df.iterrows():
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(row["Title"].lower())

        temp1 = tokenizer.tokenize(row["Title"])

        title = ' '.join(temp1)

        if exp == 2:
            raw = list(set(raw).difference(stop_words))
            row["Title"] = ' '.join([str(elem) for elem in raw])
        elif exp == 3:
            for each in raw:
                if len(each) >= 9 or len(each) <= 2:
                    raw.remove(each)

        tokenize_word(raw, title, df, index, words_removed)

    od = OrderedDict(sorted(word_freq_dict.items()))

    with open('frequency_dict.txt', 'w') as file:
        for key, val in od.items():
            file.write(str(key) + " " + str(val) + "\n")

    with open("./vocabulary.txt", "w") as file:
        for element in words_removed:
            file.write(element + "\n")

    train(od, exp)

    total_time = time.process_time() - start_time
    print('TOTAL TIME TAKEN IN (S):', total_time)
    print('TOTAL TIME TAKEN IN (MINUTES):', total_time / 60)
    print("-------------------------------------------------")


def tokenize_word(raw, title, df, index, w_removed, testing=False):
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

        indices_0 = [i for i, e in enumerate(raw) if e == word[0].lower()]
        if len(indices_0) != 0:
            indices_1 = [i for i, e in enumerate(raw[indices_0[0] + 1:]) if e == word[1].lower()]
        else:
            indices_1 = [i for i, e in enumerate(raw) if e == word[1].lower()]

        # print("RAW:", raw)
        # print(word[0], " ", word[1])

        if word[0].istitle() and word[1].istitle():
            # print("INDICES:", indices_0, " ", indices_1)
            if len(indices_0) > 0 and (
                    pos_dict.get(word[0].lower()) == 'NN' or pos_dict.get(word[0].lower()) == 'NNS'):
                if len(indices_1) > 0 and (
                        pos_dict.get(word[1].lower()) == 'NN' or pos_dict.get(word[1].lower()) == 'NNS'):
                    raw.remove(word[0].lower())
                    # raw.index(word[1].lower())
                    raw.remove(word[1].lower())

                    if testing is False:
                        temp = each_element.lower() + "-" + df.at[index, 'Post Type']
                        freq = word_freq_dict.get(temp)
                        raw.append(each_element.lower())
                        if freq is None:
                            word_freq_dict[temp] = 1
                        else:
                            freq += 1
                            word_freq_dict[temp] = freq
                    else:
                        word_list.append(each_element.lower())

    # print("AFTER:", raw)
    pos = nltk.pos_tag(raw)

    for each_word in pos:
        wordnet_tag = get_wordnet_pos(each_word[1])

        if each_word[1] == "FW" or each_word[1] == "CD":
            w_removed.add(each_word[0].strip())
            continue
        if len(each_word[0]) == 1 and not (each_word[0] == "a" or each_word[0] == "i"):
            w_removed.add(each_word[0].strip())
            continue

        word_lemm = lemmatizer.lemmatize(each_word[0], wordnet_tag)

        if testing is False:
            temp = word_lemm + "-" + df.at[index, 'Post Type']
            value = word_freq_dict.get(temp)
            if value is None:
                word_freq_dict[temp] = 1
            else:
                value += 1
                word_freq_dict[temp] = value
        else:
            if testing:
                word_list.append(word_lemm)

    pos.clear()

    return word_list


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
        freq_dict = dict(new_dict_list)
    elif exp == 5:
        smoothing = smoothing_value

    dict_keys = freq_dict.keys()
    freq = list(freq_dict.values())

    for each in dict_keys:
        word_class = each.split('-')
        word.append(word_class[0])
        post_type.append(word_class[1])

    df = pd.DataFrame({'Word': word, 'Class': post_type, 'Frequency': freq})
    df.to_csv("vocabulary.csv")

    if df.empty:
        experiments.each_accuracy = -1
        return

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

    vocabulary = df.Word.unique()
    vocabulary_size = len(vocabulary)
    experiments.no_of_words = vocabulary_size

    class_probability_show_hn = len(show_hn_dft.index) / len(df_training.index)
    class_probability_ask_hn = len(ask_hn_dft.index) / len(df_training.index)
    class_probability_poll = len(poll_dft.index) / len(df_training.index)
    class_probability_story = len(story_dft.index) / len(df_training.index)

    if smoothing == 0:
        vocabulary_size = 0

    line_count = 1

    for word in vocabulary:
        temp_show_hn_freq = show_hn_words[word] if word in show_hn_words else 0
        temp_ask_hn_freq = ask_hn_words[word] if word in ask_hn_words else 0
        temp_story_freq = story_words[word] if word in story_words else 0
        temp_poll_freq = poll_words[word] if word in poll_words else 0

        if show_hn_count == 0:
            p_word_given_show_hn = 0
        else:
            p_word_given_show_hn = ((temp_show_hn_freq + smoothing) / (show_hn_count + vocabulary_size))

        if ask_hn_count == 0:
            p_word_given_ask_hn = 0
        else:
            p_word_given_ask_hn = ((temp_ask_hn_freq + smoothing) / (ask_hn_count + vocabulary_size))

        if poll_count == 0:
            p_word_given_poll = 0
        else:
            p_word_given_poll = ((temp_poll_freq + smoothing) / (poll_count + vocabulary_size))

        if story_count == 0:
            p_word_given_story = 0
        else:
            p_word_given_story = ((temp_story_freq + smoothing) / (story_count + vocabulary_size))

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
    print("\nTime to train:", end_time)

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

    if exp == 4 or exp == 4.5 or exp == 5:
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
