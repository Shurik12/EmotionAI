# The binary is built on the host and mounted into this image, so the base must
# match the Ubuntu release used for `make build`. Building on 26.04 and running
# on a 24.04 image fails with errors such as "libspdlog.so.1.15: cannot open
# shared object file".
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt update -y && apt install -y \
    libyaml-cpp0.8 \
    libspdlog1.15 \
    libopencv-videoio410 \
    libopencv-dnn410 \
    libopencv-core410 \
    libopencv-imgproc410 \
    libopencv-imgcodecs410 \
    libfmt10 \
    libhiredis1.1.0 \
    libssl3t64 \
    libcurl4t64 \
    libpugixml1v5 \
    curl \
    ca-certificates \
    libcurlpp0t64 \
    libfftw3-double3 \
    libavcodec62 \
    libavformat62 \
    libavutil60 \
    libswresample6 \
    libswscale9 \
    && rm -rf /var/lib/apt/lists/*

# Copy shared libraries from contrib (these are not installed via apt)
COPY ./contrib/libtorch/lib/libtorch.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libtorch_cpu.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libc10.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libgomp-52f2fd74.so.1 /usr/lib/x86_64-linux-gnu/
COPY ./contrib/onnxruntime/lib64/libonnxruntime.so.1 /usr/lib/x86_64-linux-gnu/

# Update library cache
RUN ldconfig

# Set working directory
WORKDIR /emotionai

# Create directories that will be mounted
RUN mkdir -p /emotionai/logs /emotionai/uploads /emotionai/results /emotionai/storage

# Expose ports
EXPOSE 80

# Healthcheck covers both the template port (80) and the 8080 profiles.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || curl -f http://localhost:80/api/health || exit 1

# The binary will be mounted via volume, not copied
# The config will be mounted via volume
# The models will be mounted via volume
# The frontend will be mounted via volume

CMD [ "./emotionai" ]
