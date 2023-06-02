from flask import Flask, render_template, request, jsonify
import numpy as np
import praw
import time
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__, template_folder='templates')

reddit = praw.Reddit(client_id='DCW_USDBk23o37cYXu1LBQ', 
                     client_secret='DcqCmVVZybQLBvqJ7KhmR8NuXe12Vg', 
                     user_agent='ml2_project/0.0.1 (by /u/mladepet)')

# List of subreddits
subreddits = ['AskReddit', 'gaming', 'aww', 'movies', 'Showerthoughts', 'Jokes', 'science', 'books', 'Music', 'LifeProTips']

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')  # for stopword removal
nltk.download('wordnet')  # for lemmatization

def preprocess_text(text):
    # tokenize the text into words
    words = word_tokenize(text)

    # lower the case and remove punctuation
    words = [word.lower() for word in words if word.isalpha()]

    # remove words that have less than 3 characters
    words = [word for word in words if len(word) > 2]

    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # join the preprocessed words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Load the model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # index.html should contain your button and result area

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # The code that fetches a post and makes a prediction
        title_content_length = 0

        # Fetch new posts until one is found with preprocessed_title_content length > 30
        while title_content_length <= 30:
            random_subreddit = np.random.choice(subreddits)
            new_submission = reddit.subreddit(random_subreddit).hot(limit=1)

            for submission in new_submission:
                new_post = pd.DataFrame({
                    'title': [submission.title],
                    'content': [submission.selftext],
                    'upvote_ratio': [submission.upvote_ratio],
                    'created_utc': [submission.created_utc],
                    'score': [submission.score]
                })

            new_post['title_content'] = new_post['title'] + " " + new_post['content']
            clip_prompt = (new_post['title'] + " " + new_post['content']).values[0]
            new_post['preprocessed_title_content'] = new_post['title_content'].apply(preprocess_text)

            title_content_length = len(new_post['preprocessed_title_content'][0])
            time.sleep(1)

        new_post = new_post[['preprocessed_title_content', 'upvote_ratio', 'created_utc', 'score']]
        print(new_post)
       # print(new_submission.subreddit.display_name)
        predicted_subreddit_id = loaded_model.predict(new_post)
        print("Predicted subreddit name: ", subreddits[predicted_subreddit_id[0]])
        print("Effective subreddit name: ", random_subreddit)
        #parse predicted_subreddit_id to int
        predicted_subreddit_id = int(predicted_subreddit_id[0])
        print(predicted_subreddit_id)
        print(subreddits.index(random_subreddit))

        #check if predicted_subreddit_id is equal to random_subreddit
        #if yes, result = "The model predicted correctly!"
        #if no, result = "The model predicted incorrectly! - try again until it predicts it correctly"
        if predicted_subreddit_id == subreddits.index(random_subreddit):
            result = "The model predicted correctly!"
        else:
            result = "The model predicted incorrectly! - you can try again or proceed with creating a fitting picture"
       
    return jsonify({'result': result})  # return JSON

if __name__ == '__main__':
    app.run(debug=True)
