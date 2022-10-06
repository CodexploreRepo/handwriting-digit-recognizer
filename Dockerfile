FROM python:3.9 as builder
WORKDIR /digit_recognizer
COPY ./requirements-deploy.txt ./setup.py ./setup.cfg ./
RUN pip install -r requirements-deploy.txt --quiet --no-cache-dir
COPY ./src ./src
RUN pip install -e .

FROM scratch
COPY  --from=builder . .
EXPOSE 80
CMD ["uvicorn","src.backend.app.main:app","--host","0.0.0.0","--port","80"]

# COPY ./src/backend/app /src/backend/app
# COPY ./src/digit_recognizer/models /src/digit_recognizer/models
# COPY ./src/digit_recognizer/inference /src/digit_recognizer/inference
# COPY ./src/digit_recognizer/config.py /src/digit_recognizer/config.py
# COPY ./setup.py /src/setup.py
# COPY ./setup.cfg /src/setup.cfg