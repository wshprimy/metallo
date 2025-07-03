FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /workspace
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV no_proxy=

RUN echo 'alias ll="ls -la"' >> ~/.bashrc
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
COPY requirements.txt /workspace/requirements.txt
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
RUN apt-get update && \
    apt-get install -y iputils-ping net-tools iproute2 dnsutils curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install -r /workspace/requirements.txt
RUN pip install nvitop
CMD ["bash"]
