import pandas as pd
import config
import nltk
import time
from collections import OrderedDict

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def read_file():
    df = pd.read_csv("./hn2018_2019.csv")
    df["date"] = pd.to_datetime(df["Created At"])

    mask_2018 = (df["date"] > config.train_date["start"]) & (df["date"] <= config.train_date["end"])
    mask_2019 = (df["date"] > config.test_date["start"]) & (df["date"] <= config.test_date["end"])

    df_training = df.loc[mask_2018]
    df_testing = df.loc[mask_2019]

    # build_vocabulary(pd.read_csv("./sample.csv"))
    build_vocabulary(df_training)


def build_vocabulary(df):
    lemmatizer = nltk.WordNetLemmatizer()
    word_freq_dict = {}
    j = int(0)
    start_time = time.process_time()

    for title in df["Title"]:
        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)
        raw = tokenizer.tokenize(title.lower())

        bi_gram = list(nltk.bigrams(title.split()))
        pos = nltk.pos_tag(raw)
        pos_dict = dict(pos)
        bi_grams = []

        for i in bi_gram:
            bi_grams.append(("".join([w + " " for w in i])).strip())

        for each_element in bi_grams:
            word = each_element.split(" ")

            if (word[0].istitle() and word[1].istitle()):
                if word[0].lower() in raw and (
                        pos_dict.get(word[0].lower()) == 'NN' or pos_dict.get(word[0].lower()) == 'NNS'):
                    if word[1].lower() in raw and (
                            pos_dict.get(word[1].lower()) == 'NN' or pos_dict.get(word[1].lower()) == 'NNS'):
                        index1 = raw.index(word[0].lower())

                        del raw[index1]
                        index2 = raw.index(word[1].lower())

                        del raw[index2]

                        temp = each_element.lower() + "-" + df.at[j, 'Post Type']
                        freq = word_freq_dict.get(temp)

                        if freq is None:
                            word_freq_dict[temp] = 1
                        else:
                            freq += 1
                            word_freq_dict[temp] = freq

        pos = nltk.pos_tag(raw)

        for each_word in pos:
            word_net_tag = get_wordnet_pos(each_word[1])
            if each_word[1] == "FW" or each_word[1] == "CD":
                continue
            if len(each_word[0]) == 1 and not (each_word[0] == "a" or each_word[0] == "i"):
                continue

            word_lemm = lemmatizer.lemmatize(each_word[0], word_net_tag)

            temp = word_lemm + "-" + df.at[j, "Post Type"]
            value = word_freq_dict.get(temp)

            if value is None:
                word_freq_dict[temp] = 1
            else:
                value += 1
                word_freq_dict[temp] = value

        j += 1
        pos.clear()

    print('PRE PROCESSING TIME TAKEN IN (S):', time.process_time() - start_time)

    od = OrderedDict(sorted(word_freq_dict.items()))
    train(od)

    total_time = time.process_time() - start_time
    print('TOTAL TIME TAKEN IN (S):', total_time)


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

            file.write(str(line_count) + " " + str(word[0]) + " " + str(freq) + " " + str(
                temp_df_story[0]) + " " + str(
                p_word_given_story) + " " + str(temp_df_ask_hn[0]) + " " + str(
                p_word_given_ask_hn) + " " + str(
                temp_df_show_hn[0]) + " " + str(p_word_given_show_hn) + " " + str(
                temp_df_poll[0]) + " " + str(
                p_word_given_poll) + " " + '\n')

            line_count += 1


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
