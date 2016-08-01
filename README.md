# Master thesis project

Thesis title: <br>
<strong>Deep Learning Application in Ovarian Tumor Malignancy Classification</strong>

In this repository I did few things:
* preprocessed dataset (file clean_dataset.py)
* code to calculate few useful metrics (metrics.py)
* code for CV - nested k-fold approach (nested_kfold.py)
* tested old known techniques on our dataset (file cv_old_models.py)
* prepared experiment using ladder network - semi-supervised approach (file cv_ladder.py)
* prepared experiment using new deep learning techniques - supervised approach (file cv_dl.py)
* file to predict new patients using best created model (file predict.py)


## Installation

All libraries with needed version are in `requirements.txt` file.
To run experiments faster please set up flags:
```
export THEANO_FLAGS=mode=FAST_RUN,floatX=float32
```

## CV results
All results are in `./results` directory (split into ladder - semi-supervised approach and dl - supervised one).
To get best model fold by fold results just type
```
./cv_ladder.py --get-cv-results
```
or
```
./cv_dl.py --get-cv-results
```
