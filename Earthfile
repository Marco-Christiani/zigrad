VERSION 0.8

ZIG:
    FUNCTION
    ARG ZIG_VERSION=0.14.0
    ARG INSTALL_PATH=/usr/local
    ARG BINARY_LINK_PATH=/usr/local/bin
    RUN ARCH=$(uname -m) && \
        DOWNLOAD_URL="https://ziglang.org/download/${ZIG_VERSION}/zig-linux-${ARCH}-${ZIG_VERSION}.tar.xz" && \
        wget -q "$DOWNLOAD_URL" && \
        tar -xf "zig-linux-${ARCH}-${ZIG_VERSION}.tar.xz" && \
        mkdir -p "${INSTALL_PATH}" && \
        mv "zig-linux-${ARCH}-${ZIG_VERSION}" "${INSTALL_PATH}/zig" && \
        mkdir -p "${BINARY_LINK_PATH}" && \
        ln -s "${INSTALL_PATH}/zig/zig" "${BINARY_LINK_PATH}/zig" && \
        rm "zig-linux-${ARCH}-${ZIG_VERSION}.tar.xz"

deps:
    ARG PYTHON_VERSION=3.12
    # Any value from https://pytorch.org/get-started/locally/ (e.g. cpu, cu118, cu126, cu128, rocm6.3)
    ARG COMPUTE_TARGET=cpu
    FROM python:${PYTHON_VERSION}-slim-bookworm
    # Validate we got a valid compute target (e.g. cpu, cu118, cu126, cu128, rocm6.3)
    IF bash -c '[[ ! "$COMPUTE_TARGET" =~ ^(cpu|cu[0-9]{3}|rocm[0-9]{1,2}\.[0-9]{1,2})$ ]]'
        RUN echo "Invalid compute target: $COMPUTE_TARGET" && exit 1
    END
    RUN apt-get update && apt-get install -y \
        libopenblas-dev \
        build-essential \
        wget \
        && rm -rf /var/lib/apt/lists/*
    WORKDIR /app
    RUN pip3 install --break-system-packages numpy torch --index-url https://download.pytorch.org/whl/${COMPUTE_TARGET}
    DO +ZIG
    SAVE IMAGE zigrad-base:latest

build:
    ARG ENABLE_CUDA=false
    ARG ENABLE_MKL=false
    FROM +deps
    COPY --dir src scripts ./
    COPY *.zig .
    COPY *.zon .
    RUN zig build test
    RUN zig build -Doptimize=ReleaseFast -Denable_cuda=${ENABLE_CUDA} -Denable_mkl=${ENABLE_MKL}
    CMD ["./zig-out/bin/main"]
    SAVE IMAGE zigrad:latest

test:
  FROM +build
  ARG ENABLE_CUDA=false
  ARG ENABLE_MKL=false
  COPY --dir tests ./
  RUN cd tests && zig build
  RUN zig build test
  RUN ["tests/zig-out/bin/zg-test-exe"]


test-matrix:
    FROM alpine:3.18
    LET MKL_VALS="false true"
    LET CUDA_VALS="false true"
    FOR cuda IN $CUDA_VALS
      FOR mkl IN $MKL_VALS
          BUILD +test --ENABLE_CUDA=$cuda --ENABLE_MKL=$mkl
      END
    END

local-test:
    LOCALLY
    RUN source .venv/bin/activate
    RUN cd tests && zig build
    ENV PYTHONPATH=$(realpath .venv/lib/python*/site-packages)
    RUN tests/zig-out/bin/zg-test-exe
