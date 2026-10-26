# ConvoX demo page in a container. Needs an NVIDIA GPU on the host for the CUDA versions
# (docker run --gpus all ...); without one, the page still works for serial, OpenMP and MPI.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake python3 python3-pil python3-numpy python3-matplotlib \
        libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# No GPU is visible while building, so name the GPU generations to compile for.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    && cmake --build build -j"$(nproc)" \
    && python3 scripts/gen_images.py

ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EXPOSE 8000
CMD ["python3", "ui/server.py", "--host", "0.0.0.0", "--port", "8000"]
