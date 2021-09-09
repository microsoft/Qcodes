# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.192.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

ARG REMOTE_USER="vscode"
ARG PYTHON_VERSION="3.8"
RUN chown -R $REMOTE_USER /opt/conda && \
    /opt/conda/bin/conda create -n qcodes python="${PYTHON_VERSION}"

RUN /opt/conda/bin/conda init --all

SHELL ["bash", "-lc"]
RUN conda activate qcodes

RUN conda activate qcodes \
    && python -m pip install --upgrade pip setuptools wheel

# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && sudo apt-get install xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
#     && /sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1920x1200x24 -ac +extension GLX
