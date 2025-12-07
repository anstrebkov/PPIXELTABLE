FROM python:3.10-slim

# Установка системных зависимостей для обработки изображений/видео
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Настройка переменной окружения для Pixeltable
ENV PIXELTABLE_HOME=/tmp/pixeltable_home
# Отключаем токенайзеры huggingface от параллелизма (избегает дедлоков)
ENV TOKENIZERS_PARALLELISM=false

# Открываем порт и запускаем
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
