.PHONY: up down logs version-model version-model-git

MODEL ?= heart_disease_model.joblib
VERSION ?= 1.0.1
TAG_PREFIX ?= model-v

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api prometheus grafana frontend

version-model:
	python3 scripts/version_model.py --source $(MODEL) --version $(VERSION) --promote

version-model-git:
	python3 scripts/version_model.py --source $(MODEL) --version $(VERSION) --promote --git --tag-prefix $(TAG_PREFIX)
