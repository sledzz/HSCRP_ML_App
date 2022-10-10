FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PATH
CMD gunicorn --workers=4 --bind 0.0.0.0.$PATH app:app
