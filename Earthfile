VERSION 0.8

localpy:
    LOCALLY
    ENV BASE_ROOT=$(python3 -c 'import sys, pathlib; print(pathlib.Path(sys.base_prefix).resolve())')
    ENV PYVER=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
    ENV INCL=$BASE_ROOT/include/$PYVER
    ENV LIBDIR=$BASE_ROOT/lib
    ENV LIBNAME=python${PYVER#python}
    ENV PYTHONHOME=$BASE_ROOT
    ENV LD_LIBRARY_PATH=$BASE_ROOT/lib

SETUP_PYTHON_ENV:
    FUNCTION
    ENV BASE_ROOT=$(python3 -c 'import sys, pathlib; print(pathlib.Path(sys.base_prefix).resolve())')
    ENV PYVER=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
    ENV INCL=$BASE_ROOT/include/$PYVER
    ENV LIBDIR=$BASE_ROOT/lib
    ENV LIBNAME=python${PYVER#python}
    ENV PYTHONHOME=$BASE_ROOT
    ENV LD_LIBRARY_PATH=$BASE_ROOT/lib

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
    FROM python:${PYTHON_VERSION}-slim-bookworm
    RUN apt-get update && apt-get install -y \
        libopenblas-dev \
        build-essential \
        wget \
        && rm -rf /var/lib/apt/lists/*
    WORKDIR /app
    RUN pip3 install --break-system-packages numpy torch --index-url https://download.pytorch.org/whl/cpu

build-zig:
    ARG ZIGRAD_BACKEND=HOST
    FROM +deps
    DO +ZIG
    COPY --dir src scripts ./
    COPY *.zig .
    COPY *.zon .
    ENV ZIGRAD_BACKEND=${ZIGRAD_BACKEND}
    RUN zig build
    RUN mv ./zig-out/bin/main zg-main
    CMD ["./zg-main"]
    SAVE IMAGE zg:latest

test-matrix:
    FROM alpine:3.18
    LET ZIGRAD_BACKENDS="HOST CUDA"
    FOR backend IN $ZIGRAD_BACKENDS
        BUILD +test --ZIGRAD_BACKEND=$backend
    END

test:
    ARG ZIGRAD_BACKEND=HOST
    FROM +build-zig --ZIGRAD_BACKEND=${ZIGRAD_BACKEND}
    COPY --dir tests ./
    WORKDIR tests
    RUN zig build test
