import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ParseMode
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.filters import Command
from config import BOT_TOKEN
import torch
import json
from model import NeuralNet
from nltk_utils import *
import random

# Создаём телеграмм бота и выгружаем BOT_TOKEN (телеграмм токен бота) из config.py (Его нужно создать :) )
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(storage = MemoryStorage()) # Создаём диспетчер, объект из aiogram, который будет отвечать за маршрутизации и обработку входящих сообщений

# Загружаем модель из файла
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Обработчик сообщений при команде /start у бота
@dp.message(Command("start"))
async def start(message: Message):
    reply_markup = ReplyKeyboardRemove() # Убираем виртуальную клавиатуру если она была 
    await bot.send_message(chat_id=message.chat.id, text="Привет!", reply_markup=reply_markup) # Присылаем пользователю сообшение

# Обработчик событий на все сообщения, который приходят
@dp.message()
async def handle_all_messages(message: Message):

    sentence = tokenize(message.text) # Токенизируем сообщение от пользователя
    X = bag_of_words(sentence, all_words) # Используем "Мешок слов"
    X = X.reshape(1, X.shape[0]) # Изменяем форму вектора
    X = torch.from_numpy(X).to(device) # Отдаём на обработку нашему девайсу

    output = model(X) # Вычисляем выходные данные для сообщения X
    _, predicted = torch.max(output, dim=1) # Получаем категорию сообщения

    tag = tags[predicted.item()] # Находим эту категорию

    probs = torch.softmax(output, dim=1) # Вычисляем вероятность 
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75: # Если вероятность вычисленного сообщения выше 0.75, значит можно ответить пользователю
        for intent in intents['intents']:
            if tag == intent["tag"]:
                await message.answer(text = random.choice(intent['responses']))
    else:
        await message.answer("sorry...I do not understand...")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
