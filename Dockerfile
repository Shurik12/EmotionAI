FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt update -y && apt install -y \
    libyaml-cpp0.8 \
    libspdlog1.12 \
    libopencv-videoio406t64 \
    libopencv-dnn406t64 \
    libopencv-core406t64 \
    libopencv-imgproc406t64 \
    libopencv-imgcodecs406t64 \
    libfmt9 \
    libhiredis1.1.0 \
    libssl3 \
    libcurl4 \
    libpugixml1v5 \
    curl \
    ca-certificates \
    libcurlpp0t64 \
    libopencv-stitching406t64 \
    libopencv-contrib406t64 \
    libopencv-shape406t64 \
    libopencv-superres406t64 \
    libopencv-videostab406t64 \
    libopencv-viz406t64 \
    libfftw3-double3 \
    libeigen3-dev \
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

# Healthcheck using the port from config (default 80)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:80/api/health || exit 1

# The binary will be mounted via volume, not copied
# The config will be mounted via volume
# The models will be mounted via volume
# The frontend will be mounted via volume

CMD [ "./emotionai" ]