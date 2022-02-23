# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* taxifare/*.py

black:
	@black scripts/* taxifare/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr taxifare-*.dist-info
	@rm -fr taxifare.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# GCloud params
PYTHON_VERSION = 3.7
RUNTIME_VERSION = 2.1
PROJECT_ID = batch-814-815-berlin
REGION = europe-west1

# Bucket params
BUCKET_NAME = batch-814-815-berlin-bucket
BUCKET_TRAINING_FOLDER = trainings

# Package params
PACKAGE_NAME = taxifare
FILENAME = trainer

# Job params
JOB_NAME = taxifare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

# GCP directives
set_project:
	@gcloud config set project ${PROJECT_ID}

save_to_gcp:
	@echo 'Running taxifare.data...'
	@python -m taxifare.data
	@echo 'DONE'

run_locally:
	@echo 'Running ${PACKAGE_NAME}.${FILENAME}...'
	@python -m ${PACKAGE_NAME}.${FILENAME}
	@echo 'DONE'

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
	--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
	--package-path ${PACKAGE_NAME} \
	--module-name ${PACKAGE_NAME}.${FILENAME} \
	--python-version=${PYTHON_VERSION} \
	--runtime-version=${RUNTIME_VERSION} \
	--region ${REGION} \
	--stream-logs
