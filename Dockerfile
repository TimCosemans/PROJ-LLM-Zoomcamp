FROM python:3.13.4

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY data data

CMD ["streamlit", "run", "app.py"]