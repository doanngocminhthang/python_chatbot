import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# Khởi tạo các danh sách
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Xử lý dữ liệu
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tách từ trong câu mẫu ( tokenize each word )
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # Thêm tài liệu vào corpus
        documents.append((word_list, intent['tag']))

        # Thêm tag vào danh sách classes nếu chưa có
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#In kết quả ra để kiểm tra
print("Documents:", documents)
print("Classes:", classes)
print("Words:", words)
print(len(intents["intents"]))