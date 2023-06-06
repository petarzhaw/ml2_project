# Machine Learning Reddit Post Classifier

## Overview

This project uses Python and machine learning to classify the subreddit of posts based on their parameters. It fetches posts from several subreddits, preprocesses the text data, and trains several machine learning models to predict the subreddit of a post.

## Motivation
Many people use Reddit to discuss topics they are interested in. However, it can be difficult to find the right subreddit for a specific topic or they don't know which subreddit to choose for their specific post. This project aims to solve this problem by classifying subreddits based on their posts. 

## Prerequisites

The necessary libraries and their versions used in this project are listed in the `requirements.txt` file. 

## Installation

1. Clone this repository
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

This code can be run as cell by cell as an jupyter notebook (.ipynb). It fetches posts from a list of defined subreddits, stores the fetched data in a CSV file, and preprocesses the posts' text data. It then trains and evaluates multiple classifiers based on this preprocessed data.

## Project Structure

The script performs the following steps:

1. Fetch Reddit posts: It fetches the latest posts from several popular subreddits using Reddit's PRAW API.

2. Preprocess the text data: The title and content of each post are cleaned by converting to lowercase, removing punctuation, lemmatizing, and removing stopwords.

3. Vectorize the text: It transforms the preprocessed text data into TF-IDF vectors.

4. Train models: It trains multiple classifiers including SVC and Multinomial Naive Bayes. It also uses GridSearchCV to tune hyperparameters and select the best model.

5. Evaluate models: It evaluates the performance of the models using metrics like accuracy and F1-score.

6. Save and load models: The best performing model is saved to disk for future use.

7. Predict subreddit: The model is used to predict the subreddit of a new post. Test it by running the cell for the test on its own dataset. Then test it by running the cell for the api call test. You can also run the main.py file to test the model in a browser. 

## Files

- `app.ipynb`: The main Python file which fetches the data, preprocesses it, trains the models, and evaluates them.
- `requirements.txt`: Lists the Python libraries required for this project.
- `reddit.csv`: Stores the fetched Reddit posts.
- `model.pkl`: Stores the best performing model for future use.
- `main.py`: Runs the model in a browser.

## Results

The performance of the models is printed to the console in terms of accuracy, precision, recall, and F1-score. Additionally, the best performing model is saved to disk for future use.

## Acknowledgements

The Python Reddit API Wrapper (PRAW) was used to fetch data from Reddit. The Scikit-learn library was used for machine learning and text preprocessing. Pandas and NumPy were used for data handling and manipulation.

## Future Work

Future improvements may include using more advanced natural language processing techniques for preprocessing, incorporating more features into the models, and testing additional machine learning algorithms. 

## Contact

If you have any questions about this project, please feel free to reach out to me at [mladepet@students.zhaw.ch](mailto:mladepet@students.zhaw.ch)

## Additional Information

Scraping/API calls on Reddit are limited to 1 call per second and that's the reason only roughly 4000 unique samples of posts have been collected. 

This app can be further improved by adding more subreddits to the list of subreddits to fetch posts from. The more subreddits we have, the more accurate the model will be in predicting the subreddit of a post and it will be able to suggest a subreddit for user written posts.
