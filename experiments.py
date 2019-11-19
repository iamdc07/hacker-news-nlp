import pandas as pd
import nltk, math
import train
import operator
from sklearn.metrics import accuracy_score


def baseline(class_probability, df_testing, model_show_hn, model_ask_hn, model_poll, model_story):
    print('IN baseline method')
    j = df_testing.first_valid_index()

    test_labels = []
    predictions = []

    # df_testing = pd.read_csv("./sample_testing.csv")

    for index, row in df_testing.head(1000).iterrows():
        # print("Row:", row["Post Type"])
        # exit(0)
        post_type = df_testing[df_testing['Title'] == row["Title"]]['Post Type'].tolist()
        # print('Post Type:', post_type)

        # hypothesis_story = 0.0
        # hypothesis_ask_hn = 0.0
        # hypothesis_poll = 0.0
        # hypothesis_show_hn = 0.0

        # temp_story = 0.0
        # temp_ask_hn = 0.0
        # temp_show_hn = 0.0
        # temp_poll = 0.0

        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(row["Title"].lower())

        j, word_list = train.tokenize_word(raw, row["Title"], df_testing, j, True)
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

        answer = {
            "poll": hypothesis_poll,
            "show_hn": hypothesis_show_hn,
            "ask_hn": hypothesis_ask_hn,
            "story": hypothesis_story
        }

        legend = {
            "poll": 0,
            "show_hn": 1,
            "ask_hn": 2,
            "story": 3
        }

        # print("POLL:", hypothesis_poll)
        # print("Show_hn:", hypothesis_show_hn)
        # print("Ask_hn:", hypothesis_ask_hn)
        # print("Story:", hypothesis_story)

        # print(max(hypothesis_show_hn, hypothesis_ask_hn, hypothesis_poll, hypothesis_story))
        print("predicted: ['", max(answer.items(), key=operator.itemgetter(1))[0], "'] actual:", post_type, ' title:', row["Title"])
        # print(max(answer.items(), key=operator.itemgetter(1))[0])
        # print(legend.get(post_type[0]))
        test_labels.append(legend.get(post_type[0]))
        predictions.append(legend.get(max(answer.items(), key=operator.itemgetter(1))[0]))
        # exit(0)

        # int(hypothesis_poll * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_story * 10 ** 10) / 10.0 ** 10)

        # print('Bigrm:', bigrams)
        # print('Pos:', pos)
        # print('Pos1:', pos_dict)

    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)