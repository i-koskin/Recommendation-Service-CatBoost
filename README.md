# 📌 Рекомендательная система постов социальной сети

## Описание проекта

Цель проекта — разработка рекомендательной системы, которая предлагает пользователю социальной сети релевантные посты на основе:

- информации о пользователях,
- контента постов,
- истории активности пользователей.

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория

Скачайте репозиторий в локальную директорию:

```bash
git clone https://github.com/i-koskin/Recommendation-Service-CatBoost
cd Recommendation-Service-CatBoost
```

### 2. Загрузка предобученных моделей

Скачайте следующие модели и поместите их в корень проекта:

- [catboost_model_PCA](https://drive.google.com/file/d/1gksqZ9tETozRNqnV_uvKhXqSciJAXwU7/view?usp=sharing)
- [catboost_model_W2V](https://drive.google.com/file/d/1ldkckMPxD7WVJjloa97nRhXmHU8u9L_f/view?usp=sharing)

### 3. Установка зависимостей

Установите сервер для запуска API:

```bash
pip install uvicorn
```

### 4. Запуск приложения

Используйте `uvicorn` для запуска FastAPI-приложения. В зависимости от модели используйте соответствующий файл:

```bash
# Для модели на основе PCA
uvicorn app_catboost_model_PCA:app --reload

# Для модели на основе Word2Vec
uvicorn app_catboost_model_W2V:app --reload

# Для эксперимента с двумя группами ('control' и 'test')
uvicorn app:app --reload
```

> ⚠️ `app` — это имя объекта FastAPI внутри соответствующего файла (например, `app_catboost_model_PCA.py`).

---

## 📬 Работа с API

Для отправки запросов к API рекомендуем использовать [Postman](https://www.postman.com/).

### Пример запроса

- Метод: `GET`
- URL:

```http
http://localhost:8000/post/recommendations?id=200&limit=5
```

**Параметры:**

- `id` — идентификатор пользователя (`user_id`)
- `limit` — количество рекомендуемых постов

### Ответ

Пример ответа:

```json
[
  {
    "id": 100,
    "text": "Nasdaq planning $100m share sale\n\nThe owner of the technology-dominated Nasdaq stock index plans to sell shares to the public and list itself on the market it operates.\n\nAccording to a registration document filed with the Securities and Exchange Commission, Nasdaq Stock Market plans to raise $100m (£52m) from the sale. Some observers see this as another step closer to a full public listing. However Nasdaq, an icon of the 1990s technology boom, recently poured cold water on those suggestions.\n\nThe company first sold shares in private placements during 2000 and 2001. It technically went public in 2002 when the stock started trading on the OTC Bulletin Board, which lists equities that trade only occasionally. Nasdaq will not make money from the sale, only investors who bought shares in the private placings, the filing documents said. The Nasdaq is made up shares in technology firms and other companies with high growth potential. It was the most potent symbol of the 1990s internet and telecoms boom, nose-diving after the bubble burst. A recovery in the fortunes of tech giants such as Intel, and dot.com survivors such as Amazon has helped revive its fortunes.\n",
    "topic": "business"
  },
  ...
]
```

Если используется файл `app.py` (экспериментальная версия), в ответ также включается поле `exp_group`, указывающее на принадлежность пользователя к группе A/B теста:

```json
{
  "exp_group": "test",
  "recommendations": [
  {
    "id": 253,
    "text": "Venezuela and China sign oil deal\n\nVenezuelan president Hugo Chavez has offered China wide-ranging access to the countrys oil reserves.\n\nThe offer, made as part of a trade deal between the two countries, will allow China to operate oil fields in Venezuela and invest in new refineries. Venezuela has also offered to supply 120,000 barrels of fuel oil a month to China. Venezuela - the worlds fifth largest oil exporter - sells about 60% of its output to the United States. Mr Chavezs administration, which has a strained relationship with the US, is trying to diversify sales to reduce its dependence on its largest export market.\n\nChinas quick-growing economys need for oil has contributed to record-high oil prices this year, along with political unrest in the Middle East and supply bottlenecks. Oil prices are finishing the year roughly 30% higher than they were in January 2004.\n\nIn 2004, according to forecasts from the Ministry of Commerce, Chinas oil imports will be 110m tons, up 21% on the previous year. China has been a net importer of oil since the mid 1990s with more than a third of the oil and gas it consumes coming from abroad. A lack of sufficient domestic production and the need to lessen its dependence on imports from the Middle East has meant that China is looking to invest in other potential markets such as Latin America. Mr Chavez, who is visiting China, said his country would put its many of its oil facilities at the disposal of China. Chinese firms would be allowed to operate 15 mature oil fields in the east of Venezuela, which could produce more than one billion barrels, he confirmed. The two countries will also continue a joint venture agreement to produce stocks of the boiler fuel orimulsion. Mr Chavez has also invited Chinese firms to bid for gas exploration contracts which his government will offer next year in the western Gulf of Venezuela. The two countries also signed a number of other agreements covering other industries including mining.\n",
    "topic": "business"
  },
  ...
]
}
```


```json
{
  "exp_group": "control",
  "recommendations": [
  {
    "id": 6748,
    "text": "Ive seen this movie and I must say Im very impressed. There are not much movies I like, but I do like this one. You should see this movie by yourself and comment it,because this is one of my most favorite movie. I fancy to see this again. Action fused with a fantastic story. Very impressing. I like Modestys character. Actually shes very mystic and mysterious (I DO like that^^). The bad boy is pretty too. Well, actually this whole movie is rare in movieworld. I considered about the vote of this movie, I thought this is should be a very popular movie. I guess wrong. It was ME who was very impressed about this movie, and I hope Im not the only one who takes only the cost to watch this one. See and vote.",
    "topic": "movie"
  },
  ...
]
}
```

---

## 🛠 Инструменты

- [FastAPI](https://fastapi.tiangolo.com/) — backend-фреймворк
- [CatBoost](https://catboost.ai/) — модель машинного обучения
- [Postman](https://www.postman.com/) — инструмент для тестирования API
- [Uvicorn](https://www.uvicorn.org/) — сервер ASGI для запуска FastAPI

---
