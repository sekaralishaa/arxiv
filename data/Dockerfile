FROM python:3.10-slim
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir flask pandas numpy gensim scipy scikit-learn pyarrow

EXPOSE 5000
CMD ["python", "main.py"]