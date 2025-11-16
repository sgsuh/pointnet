FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && \
    apt install -y tzdata \
    libyaml-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install PyYAML==5.4.1 --ignore-installed

RUN pip install rich==13.0.0 \
    onnxsim \
    path \
    pillow==8.2.0 \
    Jinja2==3.0.3 \
    tornado==6.2 \
    pyzmq==23.0.0 \
    traitlets==5.3.0 \
    jupyter-core==4.12.0 \
    jupyter-server==1.21.0 \
    jupyter-packaging==0.10.0 \
    nbclassic==0.3.0 \
    jupytext==1.14.0 \
    markdown-it-py==2.1.0 \
    attrs==22.2.0 \
    packaging==21.3 \
    jupyterlab==3.0.0 \
    open3d==0.13.0