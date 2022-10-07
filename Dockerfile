FROM python:3.9 as builder

WORKDIR /digit_recognizer

COPY ./requirements-deploy.txt ./setup.py ./setup.cfg ./

RUN pip install -r requirements-deploy.txt --quiet --no-cache-dir

COPY ./src ./src

RUN pip install -e .

EXPOSE 80

CMD ["uvicorn","src.backend.app.main:app","--host","0.0.0.0","--port","80"]

