FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 
MAINTAINER Naoshi Kaneko <kaneko.naoshi@vss.it.aoyama.ac.jp>

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential bash-completion wget git && \
  rm -rf /var/lib/apt/lists/*

# Python setting
ENV PYTHON_VERSION=3
ENV ANACONDA_VERSION=2020.02
ENV ANACONDA_ROOT=/usr/local/anaconda
RUN cd /tmp && \
	wget https://repo.continuum.io/archive/Anaconda${PYTHON_VERSION}-${ANACONDA_VERSION}-Linux-x86_64.sh && \
	chmod +x Anaconda${PYTHON_VERSION}-${ANACONDA_VERSION}-Linux-x86_64.sh && \
        ./Anaconda${PYTHON_VERSION}-${ANACONDA_VERSION}-Linux-x86_64.sh -b -p ${ANACONDA_ROOT} && \
	rm -rf *

# PyTorch, OpenCV
RUN ${ANACONDA_ROOT}/bin/conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
RUN ${ANACONDA_ROOT}/bin/pip install opencv-contrib-python
ENV PATH=$ANACONDA_ROOT/bin:$PATH

# for OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libgtk2.0-0 libgl1-mesa-dev && \
  rm -rf /var/lib/apt/lists/*