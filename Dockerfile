# Base Image
FROM tensorflow/tensorflow:1.14.0-gpu

maintainer "Dylan Yung"

user root

# Setup environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install -y apt-utils
RUN apt-get install unzip
# RUN apt-get install -y python-tk

# Python dependencies
RUN pip install pandas
RUN pip install matplotlib
RUN pip install future-fstrings
