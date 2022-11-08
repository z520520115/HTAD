FROM python:3
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["/bin/bash"]
