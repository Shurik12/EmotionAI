FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    libyaml-cpp0.8 libspdlog1.12 \
    libopencv-videoio406t64 libopencv-dnn406t64 \
    libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
    libfmt9 libhiredis1.1.0 libssl3 libcurl4 libpugixml1v5 curl ca-certificates

RUN apt install -y libcurlpp0t64

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
    CMD curl -f http://localhost:8080/health || exit 1

CMD [ "./emotionai" ]
# CMD [ "tail", "-f", "/dev/null" ]