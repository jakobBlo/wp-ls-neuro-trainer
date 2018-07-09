#!/bin/bash

docker build -t neuro-trainer:latest .
docker run --rm -it -v "$PWD:/app" -p 1337:5000 neuro-trainer:latest bash
