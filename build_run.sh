#!/usr/bin/env bash

my_image=proj_image

# Get the short commit hash
COMMIT_HASH=$(git rev-parse --short HEAD)

# Get the current date in YYYYMMDD format
DATE=$(date +%Y%m%d)

# Combine the commit hash and date to form the BUILD_TAG
export BUILD_TAG="${DATE}_${COMMIT_HASH}"

# Build image on local Mac
docker build --platform=linux/amd64 -t ${my_image}:${BUILD_TAG} -f Dockerfile_Fastapi .

docker run -it -p 80:80 ${my_image}:${BUILD_TAG}

echo "Using image name: ${my_image}:${BUILD_TAG}"