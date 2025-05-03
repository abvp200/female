FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

# fix for NVIDIA GPG key update - https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# End of fix for NVIDIA GPGP key update (22Apr2022)

RUN apt update
# RUN apt install -y libsm6 libxext6

RUN pip install --upgrade pip

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

# RUN pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
# RUN pip install --upgrade pip
RUN pip install pycocotools

RUN python -m pip install --upgrade pip
RUN pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

# To solve problem with QtPy5 missing libs in the debug mode
# RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

RUN pip install pytorch-lightning==1.9.4

# Fix problem with jupyter
RUN pip install --upgrade notebook

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mnt