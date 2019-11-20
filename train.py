import pandas as pd
from collections import OrderedDict
import nltk, math
import experiments
import time


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
    df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]

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
            continue
        if len(each_word[0]) == 1 and not (each_word[0] == "a" or each_word[0] == "i"):
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

    if testing:
        return j, word_list
    else:
        return j


def train(freq_dict, exp):
    dict_keys = freq_dict.keys()
    freq = list(freq_dict.values())
    word = []
    word_list = []
    post_type = []
    p_ask_hn_dict = {}
    p_story_dict = {}
    p_show_hn_dict = {}
    p_poll_dict = {}
    class_probability = []

    for each in dict_keys:
        word_class = each.split('-')
        word.append(word_class[0])
        post_type.append(word_class[1])

    df = pd.DataFrame({'Word': word, 'Class': post_type, 'Frequency': freq})
    story_df = df[df.Class.str.match('story', case=False)]
    ask_hn_df = df[df.Class.str.match('ask_hn', case=False)]
    show_hn_df = df[df.Class.str.match('show_hn', case=False)]
    poll_df = df[df.Class.str.match('poll', case=False)]

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

    # int(15.55555 * 10 ** 3) / 10.0 ** 3

    print("Size:", vocabulary_size)

    class_probability_show_hn = (show_hn_count / total_words)
    class_probability_ask_hn = (ask_hn_count / total_words)
    class_probability_poll = (poll_count / total_words)
    class_probability_story = (story_count / total_words)

    # class_probability_show_hn = int(class_probability_show_hn * 10 ** 10) / 10.0 ** 10
    # class_probability_ask_hn = int(class_probability_ask_hn * 10 ** 10) / 10.0 ** 10
    # class_probability_poll = int(class_probability_poll * 10 ** 10) / 10.0 ** 10
    # class_probability_story = int(class_probability_story * 10 ** 10) / 10.0 ** 10

    line_count = 1

    for word in vocabulary:
        temp_df_show_hn_freq = show_hn_words[word] if word in show_hn_words else 0

        temp_df_ask_hn_freq = ask_hn_words[word] if word in ask_hn_words else 0

        temp_df_story_freq = story_words[word] if word in story_words else 0

        temp_df_poll_freq = poll_words[word] if word in poll_words else 0

        p_word_given_show_hn = ((temp_df_show_hn_freq + 0.5) / (show_hn_count + vocabulary_size))

        p_word_given_ask_hn = ((temp_df_ask_hn_freq + 0.5) / (ask_hn_count + vocabulary_size))

        p_word_given_poll = ((temp_df_poll_freq + 0.5) / (poll_count + vocabulary_size))

        p_word_given_story = ((temp_df_story_freq + 0.5) / (story_count + vocabulary_size))

        if exp == 1:
            file = open("model-2018.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_df_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_df_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_df_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_df_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
        elif exp == 2:
            file = open("stopword-model.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_df_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_df_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_df_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_df_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
        elif exp == 3:
            file = open("wordlength-model.txt", "a")
            file.write(str(line_count) + " " + str(word) + " " + str(temp_df_story_freq) + " " + str(
                p_word_given_story) + " " + str(
                temp_df_ask_hn_freq) + " " + str(p_word_given_ask_hn) + " " + str(
                temp_df_show_hn_freq) + " " + str(
                p_word_given_show_hn) + " " + str(temp_df_poll_freq) + " " + str(
                p_word_given_poll) + " " + '\n')
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

    experiments.baseline(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
                         p_story_dict, exp)


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
