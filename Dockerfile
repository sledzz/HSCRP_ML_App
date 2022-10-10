FROM python:3.7
COPY . /app
WORKDIR /app
ENV PORT 8000
EXPOSE $PORT
RUN pip install -r requirements.txt
CMD gunicorn --workers=4 --bind $PATH app:app
