FROM tensorflow/tensorflow:latest-devel-py3
EXPOSE 5000

WORKDIR /app

# avoid warning like: Allocation of 67108864 exceeds 10% of system memory.
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN apt update && apt install -y libsm6 libxext6  libxrender-dev python3-tk
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install simplejson
