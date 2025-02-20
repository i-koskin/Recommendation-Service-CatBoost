# Recommendation-Service-CatBoost

Проект "Рекомендательная система постов социальной сети"


Цель проекта:
Разработка рекомендательной системы, предлагающей пользователю социальной сети посты для просмотра на основании имеющихся данных о пользователе, размещенных постах, истории активности.

Стек:

• CatBoost
• Python (Pandas, Numpy, Scikit-learn, Seaborn, Scipy)
• FastAPI
• PostgreSQL, SQLAlchemy, ORM
• Git

Этапы работы над проектом
1. Сбор и подготовка данных
- Сбор данных: Исследование доступных данных о пользователях, постах и их взаимодействиях (лайки, комментарии).
- Очистка данных: Удаление дубликатов, обработка пропусков и аномалий в данных.
- Анализ данных: Первичный анализ для выявления закономерностей и интересных признаков.
- Подготовка данных: Преобразование категориальных данных методами TF-IDF, PCA и Word2Vec для последующего сравнения двух моделей.

2. Анализ и выбор алгоритма
- Изучение методов построения рекомендаций: Рассмотрение различных подходов, таких как коллаборативная фильтрация, контентная фильтрация, гибридные методы и модели на основе машинного обучения.
- Выбор алгоритма: Определение наиболее подходящего метода для решения задачи.

3. Разработка модели
- Создание модели: Реализация выбранного алгоритма и обучение модели на подготовленных данных.
- Тестирование модели**: Оценка производительности модели с помощью метрики HitRate@K (HitRate@5PCA = 0.578, HitRate@5Word2Vec = 0.598)

4. Разработка API для интеграции
- Создание RESTful API: Разработка интерфейса для взаимодействия с рекомендательной системой, позволяющего получать рекомендации для пользователей.
- Документация API: Описание всех эндпоинтов и методов взаимодействия.

5. Мониторинг
- Логирование: Реализация системы логирования для отслеживания ошибок и активности пользователей.

6. A/B тестирование
- Формирование выборки: Реализация функционала для разделения пользователей на группы.
- A/B эксперимент: Применение одной из двух реализованных моделей для построения рекомендаций для каждой из групп пользователей.
- Оценка результатов: Расчет метрик и оценка значимости изменений.

Порядок использования:
1.В терминале: uvicorn app:app --reload
- `app` — это название файла (без расширения `.py`), в котором находится ваше приложение FastAPI.
- `app` (вторая часть) — это имя объекта FastAPI внутри этого файла.
  
2.В Postman: localhost:8000/post/recommendations?id=200&time=2023-10-01T12:00:00
- `id` — user_id.
- `time=2023-10-01T12:00:00` — установленные дата/время.

  При необходимости использовать текущие дату/время выполните следующие шаги:
  2.1. Перейдите на вкладку `Params`, которая расположена под полем URL.
  2.2. В поле `value` для `time` введите следующий JavaScript-код:
     {{currentDateTime}}.
  2.3.  Перейдите на вкладку `Pre-request Script` (она находится рядом с вкладкой `Params`).
   - Вставьте следующий код:
     const currentDateTime = new Date().toISOString();
     pm.environment.set("currentDateTime", currentDateTime);

3.  Отправка запроса:
    Теперь, когда вы настроили переменные, просто нажмите кнопку `Send`, чтобы отправить запрос.
    В качестве значения:
    - для параметра `id` будет подставлен user_id пользователя, которому требуется представить рекомендованыые к просмотру посты социальной сети.
    - для параметра `time` будет подставлено установленное вами дата/время или текущие дата/время в формате ISO 8601 .    
