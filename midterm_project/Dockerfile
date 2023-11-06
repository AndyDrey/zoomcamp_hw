FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /midterm_project

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["flask_service.py", "xgb_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "flask_service:app"]