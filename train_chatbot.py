import numpy as np
from keras.models import Sequential, load_model
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

# Chuẩn hóa từ và (lemmatize), chuyển về chữ thường (lowercase) và loại bỏ các ký tự không cần thiết (ignore_letters)
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# Sort classes ( Sắp xếp classes )
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents ( documents = kết hợp giữa patterns và intents )
print(len(documents), "documents")
# classes = intents 
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data ( Tạo dữ liệu huấn luyện )
training = [] # Danh sách lưu các mẫu dữ liệu ( Bag of Words và đầu ra one-hot)
output_empty = [0] * len(classes) # Tạo một vector toàn số 0 có độ dài bằng số lượng lớp

#Duyệt qua từng tài liệu
for doc in documents:
    bag = [] # Tạo một mảng rỗng cho Bag of Words
    word_patterns = doc[0] # list  of tokenized words for the pattern
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns ] # lemmatize each word - create base word, in attempt to represent related words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) 

    # Tạo đầu ra one-hot encoding
    output_row = list(output_empty) # Sao chép vector đầu ra
    output_row[classes.index(doc[1])] = 1 # Lấy chỉ số của lớp trong danh sách classes và gán giá trị 1 cho vị trí đó
    training.append([bag, output_row]) # Thêm Bag of Words và đầu ra one-hot vào danh sách training

# Xáo trộn dữ liệu và chuyển về dạng mảng numpy
random.shuffle(training)
training = np.array(training, dtype=object)

# Tạo mảng dữ liệu đầu vào và đầu ra
train_x = [item[0] for item in training] # Bag of Words
train_y = [item[1] for item in training] # Đầu ra one-hot

#Chuyen sang Numpy array
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")

# Xây dựng model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Biên dịch model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Lưu model đã huấn luyện
model.save("chatbot_model.h5")
print("Mô hình đã được lưu.")

# Tải mô hình và dữ liệu
model = load_model("chatbot_model.h5")
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Kiểm tra câu đầu vào
print(predict_class("Hi"))
print(predict_class("Thanks!"))
print(predict_class("Bye"))

#In kết quả ra để kiểm tra
print("Documents:", documents)
print("Classes:", classes)
print("Words:", words)
print(len(intents["intents"]))