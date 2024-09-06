FROM debian:bookworm


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip python3-virtualenv python3.11-venv curl

WORKDIR /root

# install toolchain
RUN curl https://sh.rustup.rs -sSf | \
    sh -s -- --default-toolchain stable -y

ENV PATH "/root/.cargo/bin:$PATH"

RUN mkdir slow_python
COPY ./requirements.txt /root/slow_python/requirements.txt
WORKDIR /root/slow_python

ENV RUSTFLAGS "-C target-cpu=native -C target-feature=+avx2,+avx,+sse2,+avx512f,+avx512bw,+avx512vl"
ENV VIRTUAL_ENV "/root/slow_python/venv"
RUN python3 -m venv venv && \
    ./venv/bin/pip install -r requirements.txt && \
    ./venv/bin/pip install maturin

COPY . .

RUN ./venv/bin/maturin develop --release

CMD ["/root/slow_python/venv/bin/python3", "main.py"]
