FROM python:3.14
WORKDIR /code/

COPY ./requirements.txt /code/requirements.txt
COPY ./predictor /code/predictor


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "predictor.api.priceapi:app", "--host", "0.0.0.0", "--port", "80"]
