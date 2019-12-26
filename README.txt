This project is an implementation of Naive Bayes Classification on a Dataset of the website HackerNews. The dataset contains roughly 400,000 entries and each row has many fields and each row represents a post. The goal is to correctly classify a given post into one of the four classes: Ask_Hn, Show_Hn, Poll, Story. The program uses all the posts from entries of 2018 to train the model and then tests it across entries of 2019. The dataset is cleaned and bigrams are computed as an input to the training model.


The project is developed and tested with python3 with macOS v10.14.6

To run the project, install the following libraries
1. pandas
2. matplotlib
3. numpy
4. nltk
5. sklearn

----------- To execute the project from command line -----------
python3 train.py

----------- To execute the project from jupyter notebook -----------
1. Open terminal in the project directory
2. Run the following command -- "jupyter notebook"
3. Select the file from the browser -- "train.ipynb"
4. Go to cell>Run all cells on the menu bar
