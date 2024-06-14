import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
    
stemmer = PorterStemmer()

# Функция для токенизации предлоожения 
def tokenize(message):
    return nltk.word_tokenize(message)

def stem(word):
    return stemmer.stem(word.lower())

"""
bag_of_words преорбразует токенизированное предложение в вектор, в котором каждый элемент указывает на наличие 
соотвествующего слова из списка words в предложении
Этот вектор будет использован в качестве входных данных для обучения модели
"""
def bag_of_words(tokenized_sentence,words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
