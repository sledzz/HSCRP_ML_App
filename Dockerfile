FROM python:3.7
COPY . /app
WORKDIR /app
ENV PORT 8000
EXPOSE 8000
RUN pip install -r requirements.txt
CMD gunicorn --workers=4 --bind 0.0.0.0:8000 app:app
