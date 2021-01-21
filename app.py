import keras
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template

app = Flask(__name__)
model = keras.models.load_model('model/model.h5')
train = pd.read_csv('dataset/train.csv')
list_sequences_train = train["comment_text"]
max_features = 22000

tokenizer = Tokenizer(num_words=max_features)
train = tokenizer.fit_on_texts(list(list_sequences_train))

def preprocess_text(sen):
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = [x for x in request.form.values()]
    comment = [preprocess_text(comment[0])]
    
    test = tokenizer.texts_to_sequences(comment)
    final_test = pad_sequences(test, padding='post', maxlen=200)

    prediction = model.predict(final_test)
    # toxic = prediction[0,0]
    # severe_toxic = prediction[0,1]
    # obscene = prediction[0,2]
    # threat = prediction[0,3]
    # insult = prediction[0,4]
    # identity_hate = prediction[0,5]

    return render_template('index.html', prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)