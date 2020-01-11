FROM python:3.7

ADD . /app

RUN pip install -r /app/requirements.txt

WORKDIR /app

# ENV PYTHONPATH=$PYTHONPATH:/app

EXPOSE 5000

ENTRYPOINT ["python","app.py"]