FROM pangeo/pangeo-notebook:latest

RUN /bin/sh -c mamba env
RUN mamba install climlab ecco_v4_py esmf nltk openpyxl pip pooch pygeos pythia-datasets pyworld3 texttable wordcloud
RUN pip install afinn
RUN pip install pyleoclim
RUN pip install "mystatsfunctions @ https://github.com/njleach/mystatsfunctions/archive/main.zip"
RUN pip install https://github.com/mptouzel/PyDICE/archive/master.zip
USER root
RUN apt update

# Install SDFC
# To compile we need gcc so need to update apt and install (conda gcc does not work)
# Also SDFC uses legacy setup.py and not pip so have to clone and run python setup.py install and point to the eigen install
# USER root
# WORKDIR /tmp
# RUN apt update && apt install gcc g++ --yes && wget https://github.com/yrobink/SDFC-python/archive/refs/heads/main.tar.gz && tar -xzf main.tar.gz && rm -rf main.tar.gz && cd SDFC-python-main && PATH=/usr/bin:$PATH /srv/conda/envs/notebook/bin/python setup.py install eigen="/srv/conda/envs/notebook/include/eigen3"

# Switch back to home and jovyan (defaults)
WORKDIR /home/jovyan
USER jovyan
