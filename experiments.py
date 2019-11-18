import pandas as pd
import nltk, math, sys
import train


def baseline(model_df, class_probability, df_testing):
    print('IN baseline method')
    j = df_testing.first_valid_index()
    print(model_df.head(5))

    # df_testing = pd.read_csv("./sample_testing.csv")

    for title in df_testing['Title']:
        print('Sentence:', title)
        post_type = df_testing[df_testing['Title'].str.contains(title, regex=False, case=False, na=False)][
            'Post Type'].tolist()
        print('Post Type:', post_type)

        # hypothesis_story = 0.0
        # hypothesis_ask_hn = 0.0
        # hypothesis_poll = 0.0
        # hypothesis_show_hn = 0.0

        temp_story = 0.0
        temp_ask_hn = 0.0
        temp_show_hn = 0.0
        temp_poll = 0.0

        tokenizer = nltk.RegexpTokenizer(r"\w+", False, True)

        raw = tokenizer.tokenize(title.lower())

        j, word_list = train.tokenize_word(raw, title, df_testing, j, True)
        # print(word_list)

        for each_word in word_list:
            story_conditional_p = model_df[model_df['Word'].str.contains(each_word, regex=False, case=False, na=False)][
                'Story'].tolist()
            ask_hn_conditional_p = \
                model_df[model_df['Word'].str.contains(each_word, regex=False, case=False, na=False)][
                    'Ask_hn'].tolist()
            show_hn_conditional_p = \
                model_df[model_df['Word'].str.contains(each_word, regex=False, case=False, na=False)][
                    'Show_hn'].tolist()
            poll_conditional_p = model_df[model_df['Word'].str.contains(each_word, regex=False, case=False, na=False)][
                'Poll'].tolist()

            # print("Story Conditional", story_conditional_p)
            # print("Poll Conditional", poll_conditional_p)
            # print("Ask_hn Conditional", ask_hn_conditional_p)
            # print("Show_hn Conditional", show_hn_conditional_p)

            if len(story_conditional_p) != 0:
                if temp_story == 0.0:
                    temp_story = story_conditional_p[0]
                else:
                    temp_story *= story_conditional_p[0]

            if len(ask_hn_conditional_p) != 0:
                if temp_ask_hn == 0.0:
                    temp_ask_hn = ask_hn_conditional_p[0]
                else:
                    temp_ask_hn *= ask_hn_conditional_p[0]

            if len(show_hn_conditional_p) != 0:
                if temp_show_hn == 0.0:
                    temp_show_hn = show_hn_conditional_p[0]
                else:
                    temp_show_hn *= show_hn_conditional_p[0]

            if len(poll_conditional_p) != 0:
                if temp_poll == 0.0:
                    temp_poll = poll_conditional_p[0]
                else:
                    temp_poll *= poll_conditional_p[0]

        # 0: show_hn
        # 1: story
        # 2: poll
        # 3: ask_hn

        hypothesis_story = class_probability[1] * temp_story
        hypothesis_ask_hn = class_probability[3] * temp_ask_hn
        hypothesis_show_hn = class_probability[0] * temp_show_hn
        hypothesis_poll = class_probability[2] * temp_poll

        print("POLL:", hypothesis_poll)
        print("Show_hn:", hypothesis_show_hn)
        print("Ask_hn:", hypothesis_ask_hn)
        print("Story:", hypothesis_story)

        print(max(hypothesis_poll, hypothesis_story, hypothesis_ask_hn, hypothesis_show_hn))
        # exit(0)

        # int(hypothesis_poll * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_show_hn * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_ask_hn * 10 ** 10) / 10.0 ** 10)
        # int(hypothesis_story * 10 ** 10) / 10.0 ** 10)

        # print('Bigrm:', bigrams)
        # print('Pos:', pos)
        # print('Pos1:', pos_dict)
