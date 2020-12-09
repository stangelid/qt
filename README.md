# Opinion Summarization with Quantized Transformers

This repo contains the data and code for the paper:

> **Extractive Opinion Summarization in Quantized Transformer Spaces**,<br/>
> Stefanos Angelidis, Reinald Kim Amplayo, Yoshihiko Suhara, Xiaolan Wang, Mirella Lapata, <br/>
> _To appear: In Transactions of the Association for Computational Linguistics (TACL)_.<br/>
> [arXiv](https://arxiv.org/abs/2012.04443)

## The SPACE corpus

The paper introduces SPACE, a large-scale opinion summarization benchmark for
the evaluation of unsupervised summarizers.  SPACE is built on TripAdvisor
hotel reviews and includes a training set of approximately 1.1 million reviews
for over 11 thousand hotels.  For evaluation, we created a collection of
human-written, abstractive opinion summaries for 50 hotels, including
high-level general summaries and aspect summaries for six popular aspects:
_building_, _cleanliness_, _food_, _location_, _rooms_, and _service_. 

You can download the corpus from [this google drive url](https://drive.google.com/u/0/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT&export=download).

## The Quantized Transformer

The Quantized Transformer (QT) is inspired by Vector-Quantized Variational
Autoencoders, which we repurpose for popularity-driven summarization. It uses a
clustering interpretation of the quantized space and a novel extraction
algorithm to discover popular opinions among hundreds of reviews, a significant
step towards opinion summarization of practical scope. In addition, QT enables
controllable summarization without further training, by utilizing properties of
the quantized space to extract aspect-specific summaries.

## Using our model

### Setting up the environment

* __Directory structure:__ Create the necessary directories using:

		mkdir -p logs models outputs

* __Python version:__ `python3.6`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		> pip install -r requirements.txt 

* __ROUGE:__ To ensure replicability and future research, we used the original ROUGE perl implementation and the `pyrouge` wrapper. Please follow the instructions [in this guide](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/) to setup ROUGE and `pyrouge` correctly. Make sure to you have activated your conda/virtualenv environment when installing `pyrouge` 

* __SPACE training set__: The training is not included in this repo. Download SPACE via the google drive link above and copy the file `space_train.json` to the `./data/json/` directory.

### Training QT

To train QT on a subset of the training set using a GPU, go to the `./src` directory and run the following:

    python3 train.py --max_num_entities 500 --run_id run1 --gpu 0

This will run the QT model with default hyperparameters (used for general summarization in the paper), store tensorboard logs under `./logs` and save a model snapshot after every epoch under `./models` (filename: `run1_<epoch>_model.pt`).

### Summarization with QT

To perform general opinion summarization with a trained QT model, go to the `./src` directory and run the following:

	python3 extract.py --model ../models/run1_20_model.pt --run_id general_run1 --gpu 0

This will store the summaries under `./outputs/general_run1` and also the output of ROUGE evaluation in `./outputs/eval_general_run1.json`.

For aspect opinion summarization, run:

	python3 aspect_extract.py --model ../models/run1_20_model.pt --run_id aspects_run1 --gpu 0

### Using QT on a new dataset

_Instructions coming soon_ :)
