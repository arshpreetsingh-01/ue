FROM python:3.11-slim

WORKDIR /app

COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Upgrade pip first to avoid resolver issues
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]