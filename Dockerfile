FROM python:3.7
COPY . /app
WORKDIR /app
EXPOSE $PORT
RUN pip install -r requirements.txt
CMD gunicorn --bind 0.0.0.0:$PORT app:app
