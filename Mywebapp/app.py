#import libraries
from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
ps = PorterStemmer()
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

nltk.download('stopwords')


@app.route('/')
def home():
    return render_template('display.html')

def predictVar(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = vectorizer.transform([review]).toarray()
    prediction = 'REAL' if model.predict(review_vect) == 0 else 'FAKE'
    return prediction

@app.route('/predict',methods=['POST'])
def predict1():
    features=request.form['fname']
     #converting text input into numeric input for the model
    # outputMy=0
    outputMy=predictVar(features)
    # vectorizer = TfidfVectorizer()
    # features=[features]
    # vectorizer.fit(features)
    # features= vectorizer.transform(features)
    

    #predicting output using model
    # prediction = model.predict(features)
  #    return render_template('display.html',prediction="Enterd news is{}")
    # features=str(features)
    # print(type(features))
    
    # return features
    #return render_template('result_page.html',prediction=outputMy)
    return (outputMy)

# @app.route('/predict')
# def predict():
#     return "route for predict"

#things remaining to do
# how to take input using form tag
# how to convert news data numeric data understand machine learning model

if __name__ == "__main__":
    app.run(debug=True)


    # fake news
    # Darrell Lucus House Dem Aide: We Didnâ€™t Even See Comeyâ€™s Letter Until Jason Chaffetz Tweeted It
    # real news
    #  Daniel J. Flynn FLYNN: Hillary Clinton, Big Woman on Campus - Breitbart