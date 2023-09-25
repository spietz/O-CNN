#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#   see compatible tensorflow versions and cuda packages
# https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

FROM tensorflow/tensorflow:1.15.0-gpu-py3

# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update

# set working directory
WORKDIR /home

# install compile tools
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git wget gcc g++ make cpio build-essential

# vtk lib gl dependency
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev libsm6

# install python packages
RUN pip install vtk==8.* yacs protobuf==3.*

# oh my zsh
RUN apt-get install -y zsh
# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell

# update cmake
RUN apt-get -y purge --auto-remove cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get -y install cmake

# # OCNN
# RUN cd /home/ && git clone https://github.com/spietz/O-CNN.git && cd O-CNN && \
#    cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git && \
#    cd .. && mkdir build && cd build && cmake .. -DUSE_CUDA=ON && make && \
#    cd ../../tensorflow/libs && python build.py --cuda /usr/local/cuda-10.0

RUN echo "# cuda missing paths" >> /etc/bash.bashrc && \
   echo "export PATH=${PATH}:/usr/local/cuda/bin" >> /etc/bash.bashrc && \
   echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> /etc/bash.bashrc