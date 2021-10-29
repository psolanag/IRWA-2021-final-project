import re
import nltk
nltk.download('stopwords')
import json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def preprocessing(path):
    json_data = data_transform(path)
    for data in json_data:
        for col in json_data[data]:
            if type(json_data[data][col]) == str:
                json_data[data][col] = punctuation(json_data[data][col])
                json_data[data][col] = lower_case(json_data[data][col])
                json_data[data][col] = stop_words(json_data[data][col])
                json_data[data][col] = stemming(json_data[data][col])

    return json_data

def data_transform(txt):
    with open(txt, 'r') as f:
        json_data = json.loads(f.read().strip())
    return json_data

def punctuation(words):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(r'^https?:\/\/.*[\r\n]*')", "", words).split())

def lower_case(words):
    return words.lower().split()

def stop_words(words):
    word_list = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words:
            word_list.append(word)
    return word_list

def stemming(words):
    word_list = []
    for word in words:
        word = ps.stem(word)
        word_list.append(word)
    return word_list
