# MLOps demo

This repository shows the use of DVC on the [Lego mini-figure dataset](https://www.kaggle.com/ihelon/lego-minifigures-classification).

## Pre-requisite

* R (see [installation](https://cran.biotools.fr/bin/linux/ubuntu/))
* `r-base-dev`
* `libcurl4-openssl-dev`
* `libmagick++-dev`
* Python 3.8
* pipenv (`pip3 install pipenv`)

Or use docker image from [lego-mlops-runner](https://github.com/Zenika/lego-mlops-runner)

## Workspace preparation

```
Rscript -e "install.packages('renv'); renv::init(); renv::restore(); torch::install_torch()"
```

```
pipenv --python 3.8 sync
```

## Usage

```
pipenv shell
git checkout or git pull
dvc checkout # to retrieve data
dvc repro # run the pipeline
```
