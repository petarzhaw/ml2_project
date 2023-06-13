from flask import Flask, render_template, request, jsonify
import numpy as np
import praw
import time
import pandas as pd
import pickle
import nltk
import config
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__, template_folder='templates')

#Initialize a Reddit "instance" with your own credentials
reddit = praw.Reddit(client_id=config.client_id,
                     client_secret=config.client_secret,
                     user_agent=config.user_agent)

# As of June 12th, a lot of subreddits have gone private for an indefinite amount of time in protest of the upcoming
# Reddit API changes (which affects this project as well!). Due to this circumstance the list of subreddits has been
# severely cut to prevent 402 errors!

# Old list of subreddits
subreddits = ['AskReddit', 'gaming', 'aww', 'movies', 'Showerthoughts', 'Jokes', 'science', 'books', 'Music', 'LifeProTips']

# New List of subreddits - containing only non-private subs
subreddits = ['AskReddit', 'movies', 'books']

nltk.download('punkt')  #For tokenization
nltk.download('stopwords')  #For stopword removal
nltk.download('wordnet')  #For lemmatization

def preprocess_text(text):
    #Tokenize the text into words
    words = word_tokenize(text)

    #lower the case and remove punctuation
    words = [word.lower() for word in words if word.isalpha()]

    #Remove words that have less than 3 characters
    words = [word for word in words if len(word) > 2]

    #Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    #Join the preprocessed words back into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text

#Load the model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  #index.html should contain your button and result area

@app.route('/predict', methods=['POST'])
def predict():
    post_title = ''
    if request.method == 'POST':
        #The code that fetches a post and makes a prediction
        title_content_length = 0

        #Fetch new posts until one is found with preprocessed_title_content length > 30
        while title_content_length <= 30:
            random_subreddit = np.random.choice(subreddits)
            new_submission = reddit.subreddit(random_subreddit).hot(limit=1)

            for submission in new_submission:
                post_title = submission.title
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
        predicted_subreddit_id = loaded_model.predict(new_post)
        predicted_subreddit = subreddits[predicted_subreddit_id[0]]
        print("Predicted subreddit name: ", predicted_subreddit)
        print("Effective subreddit name: ", random_subreddit)
             
        #Check if predicted_subreddit is equal to random_subreddit
        if predicted_subreddit == random_subreddit:
            result = "The model predicted correctly!"
        else:
            result = "The model predicted incorrectly! - you can try again or proceed with creating a fitting picture"
    #Return JSON
    return jsonify({'result': result, 'predicted_sub': predicted_subreddit, 'actual_sub': random_subreddit, 'title': post_title})

if __name__ == '__main__':
    app.run(debug=True)
