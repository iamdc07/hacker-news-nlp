import pandas as pd
import gc
import nltk
import train
import operator
from sklearn.metrics import accuracy_score

labels = {
    "poll": 0,
    "show_hn": 1,
    "ask_hn": 2,
    "story": 3
}


def baseline(class_probability, df_testing, model_show_hn, model_ask_hn, model_poll, model_story, exp):
    gc.collect()
    print('IN baseline method')
    df_testing = pd.read_csv("./sample_testing.csv")

    test_labels, predictions, title = classify(class_probability, df_testing, model_show_hn, model_ask_hn, model_poll,
                                               model_story, exp)

    # exit(0)

    # int(hypothesis_poll * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10)
    # int(hypothesis_story * 10 ** 10) / 10.0 ** 10)

    # print('Bigrm:', bigrams)
    # print('Pos:', pos)
    # print('Pos1:', pos_dict)

    # Create and write to appropriate file

    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)


def stop_word_filtering():
    global stop_words
    # global start_time
    # word_freq_dict = {}

    stop_words_df = pd.read_csv("./Stopwords.txt")
    # data = pd.read_csv("./sample.csv")
    stop_words = stop_words_df["a"].tolist()

    # print(set_stop_words)
    # new_data = data["Title"].apply(lambda x: [item for item in x if item not in stop_words["a"].tolist()])

    # data['Title'] = data.Title.str.replace("[^\w\s]", "").str.lower()

    # data['Title'] = data['Title'].apply(lambda x: [item for item in x.split() if item not in stop_words["a"].tolist()])
    print("IN STOP WORD FILTERING")
    train.read_file(2)


def word_length_filtering():
    print("IN WORD LENGTH FILTERING")
    train.read_file(3)


def classify(class_probability, df_testing, model_show_hn, model_ask_hn, model_poll, model_story, exp):
    title = "NaN"
    title_list = []
    test_labels = []
    predictions = []

    j = df_testing.first_valid_index()
    line_count = 1
    # print(df_testing)

    for index, row in df_testing.iterrows():
        # print("Row:", row["Post Type"])
        # exit(0)
        title = row["Title"]
        post_type = row["Post Type"]
        # print('Post Type:', post_type)

        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(title.lower())

        if exp == 2:
            raw = list(set(raw).difference(stop_words))
            title = ' '.join([str(elem) for elem in raw])
        elif exp == 3:
            new_words = [word for word in list(raw) if not (len(word) >= 9 or len(word) <= 2)]
            raw = new_words
            title = ' '.join([str(elem) for elem in new_words])
        print(title)

        j, word_list = train.tokenize_word(raw, title, df_testing, j, True)
        # print(word_list)

        # 0: show_hn
        # 1: ask_hn
        # 2: poll
        # 3: story

        hypothesis_story = class_probability[3]
        hypothesis_ask_hn = class_probability[1]
        hypothesis_show_hn = class_probability[0]
        hypothesis_poll = class_probability[2]

        # print("RAW:", raw)

        for each_word in word_list:
            # print("WORD:", each_word)
            # print(model_story["Word"].head(2))
            # temp_df_story = story_df[story_df['Word'] == word]["Frequency"].tolist()
            # print(model_story[model_story['Word'] == each_word])
            p_conditional_story = model_story[model_story['Word'] == each_word]['Story'].tolist()
            p_conditional_ask_hn = model_ask_hn[model_ask_hn['Word'] == each_word]['Ask_hn'].tolist()
            p_conditional_show_hn = model_show_hn[model_show_hn['Word'] == each_word]['Show_hn'].tolist()
            p_conditional_poll = model_poll[model_poll['Word'] == each_word]['Poll'].tolist()

            # print("Story Conditional", p_conditional_story)
            # print("Poll Conditional", p_conditional_poll)
            # print("Ask_hn Conditional", p_conditional_ask_hn)
            # print("Show_hn Conditional", p_conditional_show_hn)

            if len(p_conditional_story) != 0:
                hypothesis_story *= p_conditional_story[0]
            if len(p_conditional_ask_hn) != 0:
                hypothesis_ask_hn *= p_conditional_ask_hn[0]
            if len(p_conditional_show_hn) != 0:
                hypothesis_show_hn *= p_conditional_show_hn[0]
            if len(p_conditional_poll) != 0:
                hypothesis_poll *= p_conditional_poll[0]

            # del p_conditional_story
            # del p_conditional_ask_hn
            # del p_conditional_poll
            # del p_conditional_show_hn

        answer = {
            "poll": hypothesis_poll,
            "show_hn": hypothesis_show_hn,
            "ask_hn": hypothesis_ask_hn,
            "story": hypothesis_story
        }

        # print("POLL:", hypothesis_poll)
        # print("Show_hn:", hypothesis_show_hn)
        # print("Ask_hn:", hypothesis_ask_hn)
        # print("Story:", hypothesis_story)

        # print(max(hypothesis_show_hn, hypothesis_ask_hn, hypothesis_poll, hypothesis_story))
        prediction = max(answer.items(), key=operator.itemgetter(1))[0]
        print("predicted: ['", prediction, "'] actual:", post_type, ' title:',
              title)
        # print(labels.get(post_type[0]))
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
