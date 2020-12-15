FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt ./requirements.txt

RUN apt-get update && \ 
    pip install -r requirements.txt  && \
    conda install opencv --channel conda-forge

COPY . .

ENV PYTHONUNBUFFERED=1

#CMD ["python3", "-u", "app.py"]