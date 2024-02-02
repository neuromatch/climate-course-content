FROM pangeo/pangeo-notebook:latest
# FROM ubuntu:22.04

# RUN python --version



COPY docker-environment.yml /tmp/environment.yml
# RUN /bin/sh -c mamba env
# USER root
# WORKDIR /tmp
# RUN /bin/sh -c apt update

RUN mamba env update -f /tmp/environment.yml -n notebook
# RUN mamba env update -f /tmp/environment.yml -n climatematch

# Install SDFC
# To compile we need gcc so need to update apt and install (conda gcc does not work)
# Also SDFC uses legacy setup.py and not pip so have to clone and run python setup.py install and point to the eigen install
# USER root
# WORKDIR /tmp
# RUN apt update && apt install gcc g++ --yes && wget https://github.com/yrobink/SDFC-python/archive/refs/heads/main.tar.gz && tar -xzf main.tar.gz && rm -rf main.tar.gz && cd SDFC-python-main && PATH=/usr/bin:$PATH /srv/conda/envs/notebook/bin/python setup.py install eigen="/srv/conda/envs/notebook/include/eigen3"

# Switch back to home and jovyan (defaults)
WORKDIR /home/jovyan
USER jovyan
