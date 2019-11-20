import pandas as pd
import gc
import nltk
import train
import operator, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

no_of_words = 0
each_accuracy = 0

labels = {
    "poll": 0,
    "show_hn": 1,
    "ask_hn": 2,
    "story": 3
}

plt.rcdefaults()


def baseline(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
             p_story_dict, exp):
    gc.collect()
    print('IN baseline method')
    df_testing = pd.read_csv("./sample_testing.csv")

    test_labels, predictions, title = classify(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict,
                                               p_poll_dict,
                                               p_story_dict, exp)

    # int(hypothesis_poll * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_story * 10 ** 10) / 10.0 ** 10)

    # Calculate precision and recall and plot it against the data

    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)
    print("--------------------------------------")
    return accuracy


def stop_word_filtering():
    global stop_words

    stop_words_df = pd.read_csv("./Stopwords.txt")
    stop_words = stop_words_df["a"].tolist()

    print("IN STOP WORD FILTERING")
    train.read_file(2)


def word_length_filtering():
    print("IN WORD LENGTH FILTERING")
    train.read_file(3)


def infrequent_word_filtering():
    # global vocab_size
    # global accuracy_list
    vocab_size = []
    accuracy_list = []

    i = 5
    print("IN INFREQUENT WORD FILTERING")

    train.read_file(4)
    vocab_size.append(no_of_words)
    accuracy_list.append(each_accuracy)

    while i <= 20:
        train.remove_freq = i
        train.read_file(4)
        vocab_size.append(no_of_words)
        accuracy_list.append(each_accuracy)
        i += 5

    # print("Vocab:", vocab_size)
    # print("Accuracy:", accuracy_list)

    objects = ('=1', '<=5', '<=10', '<=15', '<=20')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, accuracy_list, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, objects)
    plt.xlabel('Frequency')
    plt.ylabel("Accuracy")
    plt.title('Performance of the classifiers against the number of words ')

    plt.show()

    i = 5
    vocab_size.clear()
    accuracy_list.clear()

    while i <= 25:
        train.remove_percent = i/100
        train.read_file(4.5)
        vocab_size.append(no_of_words)
        accuracy_list.append(each_accuracy)
        i += 5

    objects = ('5%', '10%', '15%', '20%', '25%')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, accuracy_list, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, objects)
    plt.xlabel('Frequent words')
    plt.ylabel("Accuracy")
    plt.title('Performance of the classifiers against the number of words ')

    plt.show()

def smoothing():
    print("IN SMOOTHING")
    accuracy_list = []


def classify(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
             p_story_dict, exp):
    title = "NaN"
    title_list = []
    test_labels = []
    predictions = []

    j = df_testing.first_valid_index()
    line_count = 1

    for index, row in df_testing.iterrows():
        title = row["Title"]
        post_type = row["Post Type"]
        # print(line_count, " ", title)

        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(title.lower())

        # if exp == 2:
        #     raw = list(set(raw).difference(stop_words))
        #     title = ' '.join([str(elem) for elem in raw])
        if exp == 3:
            new_words = [word for word in list(raw) if not (len(word) >= 9 or len(word) <= 2)]
            raw = new_words
            title = ' '.join([str(elem) for elem in new_words])

        j, word_list = train.tokenize_word(raw, title, df_testing, j, True)

        # 0: show_hn
        # 1: ask_hn
        # 2: poll
        # 3: story

        hypothesis_story = math.log10(class_probability[3])
        hypothesis_ask_hn = math.log10(class_probability[1])
        hypothesis_show_hn = math.log10(class_probability[0])
        # print(class_probability[0])
        # print(class_probability[2])
        hypothesis_poll = math.log10(class_probability[2])

        for each_word in word_list:

            if each_word in p_story_dict:
                p_conditional_story = p_story_dict[each_word]
                hypothesis_story += math.log10(p_conditional_story)
                hypothesis_story = int(hypothesis_story * 10 ** 10) / 10.0 ** 10

            if each_word in p_ask_hn_dict:
                p_conditional_ask_hn = p_ask_hn_dict[each_word]
                hypothesis_ask_hn += math.log10(p_conditional_ask_hn)
                hypothesis_ask_hn = int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10

            if each_word in p_show_hn_dict:
                p_conditional_show_hn = p_show_hn_dict[each_word]
                hypothesis_show_hn += math.log10(p_conditional_show_hn)
                hypothesis_show_hn = int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10

            if each_word in p_poll_dict:
                p_conditional_poll = p_poll_dict[each_word]
                hypothesis_poll += math.log10(p_conditional_poll)
                hypothesis_poll = int(hypothesis_poll * 10 ** 10) / 10.0 ** 10

        answer = {
            "poll": hypothesis_poll,
            "show_hn": hypothesis_show_hn,
            "ask_hn": hypothesis_ask_hn,
            "story": hypothesis_story
        }

        prediction = max(answer.items(), key=operator.itemgetter(1))[0]
        # print("predicted: ['", prediction, "'] actual:", post_type, ' title:',
        #       title)
        title_list.append(title)
        test_labels.append(labels.get(post_type))
        predictions.append(labels.get(max(answer.items(), key=operator.itemgetter(1))[0]))

        if exp == 1:
            file = open("baseline-result.txt", "a")
            file.write(str(line_count) + " " + title + " " + prediction + " " + str(
                hypothesis_story) + " " + str(
                hypothesis_ask_hn) + " " + str(hypothesis_show_hn) + " " + str(
                hypothesis_poll) + " " + str(
                post_type) + " " + ("right" if post_type == prediction else "wrong") + '\n')
            file.close()
        elif exp == 2:
            file = open("stopword-result.txt", "a")
            file.write(str(line_count) + " " + title + " " + prediction + " " + str(
                hypothesis_story) + " " + str(
                hypothesis_ask_hn) + " " + str(hypothesis_show_hn) + " " + str(
                hypothesis_poll) + " " + str(
                post_type) + " " + ("right" if post_type == prediction else "wrong") + '\n')
            file.close()
        elif exp == 3:
            file = open("wordlength-result.txt", "a")
            file.write(str(line_count) + " " + title + " " + prediction + " " + str(
                hypothesis_story) + " " + str(
                hypothesis_ask_hn) + " " + str(hypothesis_show_hn) + " " + str(
                hypothesis_poll) + " " + str(
                post_type) + " " + ("right" if post_type == prediction else "wrong") + '\n')
            file.close()
        line_count += 1

    return test_labels, predictions, title


def select_experiment():
    user_input = 0

    while user_input != -1:
        print("Choose your experiment")
        print("2. Stopwords")
        print("3. Word length Filtering")
        print("4. Infrequent Word Filtering")
        print("5. Smoothing\n")
        print("Type '-1' to exit")
        user_input = int(input("Enter your choice:"))

        if user_input == 2:
            stop_word_filtering()
        elif user_input == 3:
            word_length_filtering()
        elif user_input == 4:
            infrequent_word_filtering()
        elif user_input == 5:
            smoothing()
