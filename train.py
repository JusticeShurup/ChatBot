import numpy as np
import json
import torch
import torch.nn as nn
from nltk import download 
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

"""
intents.json файл содержит заготовленные реплики, которыми бот будет оперировать
Структура файла такова
intents.json содержит в себе массив intents, который в свою очередь содержит объекты с полями
tag - Который является категорией ответов
patterns - Пример предложения на который бот должен реагировать
responses - Заготовленные ответы - то как бот будет отвечать
"""

# Считываем json 
with open('intents.json', 'r') as f: 
    intents = json.load(f)

# Создаём заранее массив для всех слов, тегов и пар<вопрос, ответ>
all_words = []
tags = []
pairs = []

# Парсим расспаршенный json файл :)
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        pairs.append((w, tag))

ignore_words = ['?', '.', '!'] # Игнорируем знаки препинания, они для нас не несут смысловой нагрузки

"""
    Используем алгоритм стемминга, это необходимо для вычленения из предложения и разных форм слов основной сути
    Так например из слова playing алгоритм возьмёт лишь слово play, для передачи основой сути, а также экономии места
"""
all_words = [stem(w) for w in all_words if w not in ignore_words] # Используем алгоритм Стемминга

all_words = sorted(set(all_words)) # Сортируем все слова, а также убираем повторения, с помощью преобразования массива во множество
tags = sorted(set(tags)) # Проделываем тоже самое с тегами

# Выводим информацию по данным, по которым будет обучаться бот
print(len(pairs), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Подготоваливаем данные для обучения с помощью "Мешка слов"
X_train = []
y_train = []
for (pattern_sentence, tag) in pairs:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    print(bag, label)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

"""
Задаём количество эпох, размер батча, размер шага, размерность входного слоя, размер скрытого слоя и 
и размер выходного слоя
"""
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

"""
Создаём класс ChatDataset, который будет являться наследников класса Dataset из библиотеки torch
переопределяем абстрактные методы родителя, используем наши данные для обучения
"""
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
# Создаём DataLoader, который будет отвечать за итеративную выборку батчей образов из нашего обучающего набора
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Если на компьютере, на котором запущенна данная программа есть видеокарта с CUDA ядрами, то
# выбираем её, чтобы вычисления производились на видеокарте, а не на процессоре, это ускорит процесс обучения
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создаём языковую модель и передаём ей наш девайс на котором будут производиться вычисления
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Создаём объект функции потерть, который будет использоваться для вычисления ошибки между 
# предсказанными данными и выходными данными модели
criterion = nn.CrossEntropyLoss() 

# Создаём объект оптимизатора Adam, который будет использоваться для обновления весов модели
# во время обучения
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Обучаем модель
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        

        outputs = model(words)
        loss = criterion(outputs, labels)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Выводим данные 
print(f'final loss: {loss.item():.4f}')

# Создаём модель
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

# Выгружаем модель в файл
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
