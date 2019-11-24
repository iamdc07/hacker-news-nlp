import pandas as pd
import gc
import nltk
import train
import operator, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    test_labels, predictions, title = classify(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict,
                                               p_poll_dict,
                                               p_story_dict, exp)

    # print(test_labels, " ", predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Measure:", f1)
    return accuracy


def stop_word_filtering():
    global stop_words

    stop_words_df = pd.read_csv("./Stopwords.txt")
    stop_words = stop_words_df["a"].tolist()

    train.read_file(2)


def word_length_filtering():
    train.read_file(3)


def infrequent_word_filtering():
    vocab_size = []
    accuracy_list = []

    i = 5

    print("Remove words which have:")
    print("Frequency = ", 1)
    train.read_file(4)
    vocab_size.append(no_of_words)
    accuracy_list.append(each_accuracy)

    while i <= 20:
        print("Frequency <= ", i)
        train.remove_freq = i
        train.read_file(4)
        vocab_size.append(no_of_words)
        if each_accuracy == -1:
            print("There is no data in dataset to perform this experiment!\n")
            return

        accuracy_list.append(each_accuracy)
        i += 5

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
        print(i, "% Frequent words")
        train.remove_percent = i / 100
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
    accuracy_list = []
    smoothing_list = []

    i = 0

    while i <= 1:
        print("Smoothing: ", format(i, '.1g'))
        train.smoothing_value = i
        train.read_file(5)
        accuracy_list.append(each_accuracy)
        smoothing_list.append(i)
        i = i + 0.1

    objects = ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, accuracy_list, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, objects)
    plt.xlabel('Smoothing')
    plt.ylabel("Accuracy")
    plt.title('Performance of the classifiers against the variation in smoothing ')

    plt.show()


def classify(class_probability, df_testing, p_show_hn_dict, p_ask_hn_dict, p_poll_dict,
             p_story_dict, exp):
    title = "NaN"
    title_list = []
    test_labels = []
    predictions = []

    line_count = 1
    # print("TEST")

    for index, row in df_testing.iterrows():
        title = row["Title"]
        post_type = row["Post Type"]
        # print("Test1:", title)

        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(title.lower())

        if exp == 3:
            new_words = [word for word in list(raw) if not (len(word) >= 9 or len(word) <= 2)]
            raw = new_words
            title = ' '.join([str(elem) for elem in new_words])

        word_list = train.tokenize_word(raw, title, df_testing, index, set(), True)

        # 0: show_hn
        # 1: ask_hn
        # 2: poll
        # 3: story

        if class_probability[3] == 0:
            hypothesis_story = 0
        else:
            hypothesis_story = math.log10(class_probability[3])

        if class_probability[1] == 0:
            hypothesis_ask_hn = 0
        else:
            hypothesis_ask_hn = math.log10(class_probability[1])

        if class_probability[0] == 0:
            hypothesis_show_hn = 0
        else:
            hypothesis_show_hn = math.log10(class_probability[0])

        if class_probability[2] == 0:
            hypothesis_poll = 0
        else:
            hypothesis_poll = math.log10(class_probability[2])

        for each_word in word_list:

            if each_word in p_story_dict:
                p_conditional_story = p_story_dict[each_word]
                if p_conditional_story != 0:
                    hypothesis_story += math.log10(p_conditional_story)
                    hypothesis_story = int(hypothesis_story * 10 ** 10) / 10.0 ** 10

            if each_word in p_ask_hn_dict:
                p_conditional_ask_hn = p_ask_hn_dict[each_word]
                if p_conditional_ask_hn != 0:
                    hypothesis_ask_hn += math.log10(p_conditional_ask_hn)
                    hypothesis_ask_hn = int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10

            if each_word in p_show_hn_dict:
                p_conditional_show_hn = p_show_hn_dict[each_word]
                if p_conditional_show_hn != 0:
                    hypothesis_show_hn += math.log10(p_conditional_show_hn)
                    hypothesis_show_hn = int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10

            if each_word in p_poll_dict:
                p_conditional_poll = p_poll_dict[each_word]
                if p_conditional_poll != 0:
                    hypothesis_poll += math.log10(p_conditional_poll)
                    hypothesis_poll = int(hypothesis_poll * 10 ** 10) / 10.0 ** 10

        temp_dict = {
            "poll": hypothesis_poll,
            "show_hn": hypothesis_show_hn,
            "ask_hn": hypothesis_ask_hn,
            "story": hypothesis_story
        }

        answer = {x: y for x, y in temp_dict.items() if y != 0}

        prediction = max(answer.items(), key=operator.itemgetter(1))[0]
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
