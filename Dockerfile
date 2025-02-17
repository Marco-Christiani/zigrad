FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH="/root/.zig/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64:${LIBRARY_PATH}"
ENV CPATH="/opt/intel/mkl/include:/usr/local/cuda/include:${CPATH}"
ENV CUDA_HOME="/usr/local/cuda"
# ENV DEBIAN_FRONTEND=noninteractive \
#     TZ=Etc/UTC \
#     PATH="/root/.zig/bin:${PATH}" \
#     LD_LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
#     LIBRARY_PATH="/opt/intel/oneapi/mkl/latest/lib/intel64:/usr/local/cuda/lib64:${LIBRARY_PATH}" \
#     CPATH="/opt/intel/oneapi/mkl/latest/include:/usr/local/cuda/include"

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gpg-agent \
    curl \
    git \
    build-essential \
    ninja-build \
    unzip \
    pkg-config \
    libssl-dev \
    libtbb-dev \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Zig
RUN wget -q https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz && \
    tar -xf zig-linux-x86_64-0.13.0.tar.xz && \
    mv zig-linux-x86_64-0.13.0 /root/.zig && \
    ln -s /root/.zig/zig /usr/local/bin/zig && \
    rm zig-linux-x86_64-0.13.0.tar.xz

# MKL (oneapi)
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-essentials-os=linux&dl-lin=apt
# RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
#     gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
#     echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
#     apt-get update && \
#     apt-get install -y intel-deep-learning-essentials && \
#     rm -rf /var/lib/apt/lists/*
# MKL on its own bc why doesnt the oneapi install play nice im not sure rn
RUN wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-mkl && \
    rm -rf /var/lib/apt/lists/*

# cutensor, cudnn
RUN apt-get update && apt get install -y \
    libcutensor2 \
    libcutensor2-dev \
    cudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# CMake.
# ubuntu cmake out of date (shocker) and wont support the "native" option for detecting the gpu compute arch
RUN wget -qO- "https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local


# RUN echo 'options nvidia "NVreg_RestrictProfiling=0"' >> /etc/modprobe.d/nvidia.conf

WORKDIR /app

COPY . .

CMD ["/bin/bash"]
