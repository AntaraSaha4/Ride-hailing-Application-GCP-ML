#!/usr/bin/env bash

# A name should start with a letter and contain only letters, numbers and underscores.
JOB_NAME="antarasa_$(date +"%m%d_%H%M")"
REGION=us-east1
CONFIG_YAML=config.yaml

# https://cloud.google.com/sdk/gcloud/reference/ai/hp-tuning-jobs/create
gcloud ai hp-tuning-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG_YAML}" \
