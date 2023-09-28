#! /bin/bash

docker buildx build -t us-west1-docker.pkg.dev/ml-elasticity/elastic-ml/varuna .
docker push us-west1-docker.pkg.dev/ml-elasticity/elastic-ml/varuna
