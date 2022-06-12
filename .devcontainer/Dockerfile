FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#   see compatible tensorflow versions and cuda packages
# https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible

# set working directory
WORKDIR /home

# install compile tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nano git gcc g++ make cmake wget cpio build-essential

# vtk lib gl dependency
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev

# python3.7 via conda
# download and install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN cd /tmp && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ./miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p $CONDA_DIR

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# update
RUN conda update -n base -c defaults conda

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
    
# install packages (conda tensorflow-gpu cannot see gpu, so we need to use pip)
RUN conda install -c conda-forge pip vtk==8.1.2 yacs
RUN pip install cmake protobuf==3.20 tensorflow-gpu==1.14.0 

# oh my zsh
RUN apt-get install -y zsh
# Uses "robbyrussell" theme (original Oh My Zsh theme), with no plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell

#RUN apt-get -y purge --auto-remove cmake
#RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
#RUN apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ bionic main'
#RUN apt-get -y cmake

## OCNN
#RUN cd /home/ && git clone https://github.com/spietz/O-CNN.git && \
#    cd O-CNN && git checkout lm_detect && \
#    cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git && \
#    cd .. && mkdir build && cd build && cmake .. -DUSE_CUDA=ON && make && \
#    cd ../../tensorflow/libs && python build.py --cuda /usr/local/cuda-10.0

#make conda activate command available from /bin/bash --login shells
#RUN echo "# cuda missing paths" >> /etc/bash.bashrc && \
#    echo "export PATH=${PATH}:/usr/local/cuda/bin" >> /etc/bash.bashrc && \
#    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> /etc/bash.bashrc
