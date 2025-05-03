# ARG TF_VERSION=2.1.0
# FROM tensorflow/tensorflow:${TF_VERSION}-gpu-pjupyter
# ARG TF_VERSION=2.8.0
# FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3-jupyter

# FROM tensorflow/tensorflow:2.8.2-gpu-jupyter
# FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt update

#docker pull tensorflow/tensorflow:2.8.0-gpu-jupyter
# FROM msranni/nni

# RUN pip install autokeras

#FROM garawalid/autokeras:latest
##FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN pip install --upgrade pip

# COPY requirements.txt .
# RUN pip install -r requirements.txt

RUN pip install click Sphinx coverage awscli flake8 python-dotenv>=0.5.1 ediblepickle hyperopt matplotlib seaborn pandas numpy scikit-learn jupyter xlrd imutils dpath pillow cmake PyQt5 ipyparallel graphviz==0.8.4 nni ImageHash

RUN ln -fs /usr/share/zoneinfo/Europe/Riga /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt install -y git
RUN export GIT_PYTHON_REFRESH=quiet


WORKDIR /opt/build

ENV OPENCV_VERSION="4.5.1"

# RUN apt-get install -y snapd
# RUN snap install libtiff-dev libavformat-dev

# RUN sed -i 's|http://us.|http://|g' /etc/apt/sources.list
# RUN sed -E -i 's#http://[^\s]*archive\.ubuntu\.com/ubuntu#http://be.archive.ubuntu.com/ubuntu#g' /etc/apt/sources.list

# RUN echo "nameserver 8.8.8.8" | tee /etc/resolv.conf > /dev/null
RUN apt-get update

RUN apt-get -qq install -y --no-install-recommends --fix-missing \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        # libopenjp2-7-de \
        libavformat-dev \
        libpq-dev

RUN pip install opencv-python-headless==4.6.0.66
RUN apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY src/requirements.txt .
# RUN pip install -r requirements.txt

RUN python -m pip install --upgrade pip
# RUN pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
# RUN pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

# # To solve problem with QtPy5 missing libs in the debug mode
# RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

RUN pip install pytorch-lightning==1.9.4

RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libpulse0 \
    libxrender1 \
    libxext6 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-glx0 \
    libxcb-xinerama0

ENV QT_QPA_PLATFORM=offscreen


WORKDIR /mnt