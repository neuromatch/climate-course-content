# FROM pangeo/pangeo-notebook:latest
FROM pangeo/pangeo-notebook:2023.07.02

COPY environment.yml /tmp/environment.yml
# RUN conda config --set channel_priority flexible
# RUN conda config --set solver classic 
RUN mamba env update -f /tmp/environment.yml -n notebook
USER root
RUN apt update

# Used to fixed issue caused by conda: https://github.com/pypa/setuptools/issues/4747#issuecomment-2552972740
RUN pip install backports.tarfile

# Install SDFC
# To compile we need gcc so need to update apt and install (conda gcc does not work)
# Also SDFC uses legacy setup.py and not pip so have to clone and run python setup.py install and point to the eigen install
# USER root
# WORKDIR /tmp
# RUN apt update && apt install gcc g++ --yes && wget https://github.com/yrobink/SDFC-python/archive/refs/heads/main.tar.gz && tar -xzf main.tar.gz && rm -rf main.tar.gz && cd SDFC-python-main && PATH=/usr/bin:$PATH /srv/conda/envs/notebook/bin/python setup.py install eigen="/srv/conda/envs/notebook/include/eigen3"

# Switch back to home and jovyan (defaults)
WORKDIR /home/jovyan
USER jovyan
