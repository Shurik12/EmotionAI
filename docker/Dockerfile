# Must match the Ubuntu release used to build the host binary mounted into the
# image (see the root Dockerfile).
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    libyaml-cpp0.8 libspdlog1.15 \
    libopencv-videoio410 libopencv-dnn410 \
    libopencv-core410 libopencv-imgproc410 libopencv-imgcodecs410 \
    libfmt10 libhiredis1.1.0 libssl3t64 libcurl4t64 \
    libpugixml1v5 libcurlpp0t64 libfftw3-double3 \
    libavcodec62 libavformat62 libavutil60 libswresample6 libswscale9 \
    curl ca-certificates

COPY ./contrib/libtorch/lib/libtorch.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libtorch_cpu.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libc10.so /usr/lib/x86_64-linux-gnu/
COPY ./contrib/libtorch/lib/libgomp-52f2fd74.so.1 /usr/lib/x86_64-linux-gnu/
COPY ./contrib/onnxruntime/lib64/libonnxruntime.so.1 /usr/lib/x86_64-linux-gnu/

# Update library cache
RUN ldconfig

WORKDIR /emotionai

COPY ./contrib/emotiefflib/models/emotieffcpplib_prepared_models/* ./models/
COPY ./contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/data/models/* ./models/

# Expose ports
EXPOSE 8080 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || curl -f http://localhost:8081/api/health || exit 1

CMD [ "./emotionai" ]
# CMD [ "tail", "-f", "/dev/null" ]
