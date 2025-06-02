FROM python:3.10-slim

WORKDIR /app

# Salin hanya file yang dibutuhkan
COPY . .

# Install dependencies secara lebih stabil
RUN pip install --no-cache-dir streamlit requests

EXPOSE 8501

# Jalankan Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

