# Opinion Summarization with Quantized Transformers

This repository contains the data and code for paper:

> **Extractive Opinion Summarization in Quantized Transformer Spaces**,<br/>
> Stefanos Angelidis, Reinald Kim Amplayo, Yoshihiko Suhara, Xiaolan Wang, Mirella Lapata, <br/>
> _To appear: In Transactions of the Association for Computational Linguistics (TACL)_.<br/>
> [arXiv](https://arxiv.org/abs/2012.04443)



## The SPACE corpus

<img align="right" src="http://homepages.inf.ed.ac.uk/sangelid/space_logo2.png" width="250px"/>

The paper introduces SPACE, a large-scale opinion summarization benchmark for
the evaluation of unsupervised summarizers.  SPACE is built on TripAdvisor
hotel reviews and includes a training set of approximately 1.1 million reviews
for over 11 thousand hotels.  

For evaluation, we created a collection of human-written, abstractive opinion
summaries for 50 hotels, including high-level general summaries and aspect
summaries for six popular aspects: _building_, _cleanliness_, _food_,
_location_, _rooms_, and _service_. Every summary is based on 100 input
reviews, an order of magnitude increase compared to existing corpora. In total,
SPACE contains 1,050 gold standard summaries. You can view the full
instructions for out multi-stage annotation procedure
[here](https://github.com/stangelid/qt/blob/main/annotation.md).

__Download the corpus from [this google drive url](https://drive.google.com/u/0/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT&export=download)__.

Here is an example drawn from SPACE of one general and 3 aspect-specific
summaries for the same hotel:

|__Hotel Vintage Park__|
|-----------------------------------------------|
|__General:__ The staff were all very friendly, helpful, and attentive. The cleanliness of the hotel was super. The rooms had a cozy elegance but garish carpets and okay beds. The Tulio restaurant was amazing, but the room service and breakfast, while prompt, were average. The hotel is within five blocks of the Pike Street Market and other upscale shopping districts.|
|__Food:__ Hotel Vintage Park has incredibly delicious, relatively inexpensive room service and delicious coffee starting at 5:00 am. A highlight was the free evening wine tasting.|
|__Service:__ All the professional staff was very helpful, attentive, and friendly, accommodating customers around every corner with kindness, courtesy, and pleasantry. They're also very pet friendly.|
|__Location:__ The location is within perfect walking distance to downtown Seattle, Pioneer Square, Pike's Market, the Waterfront, the 5th Avenue theatre, and the Westlake mall.|

## The Quantized Transformer

<img align="right" src="http://homepages.inf.ed.ac.uk/sangelid/qt_logo.png"/>

The Quantized Transformer (QT) is inspired by Vector-Quantized Variational
Autoencoders, which we repurpose for popularity-driven summarization. It uses a
clustering interpretation of the quantized space and a novel extraction
algorithm to discover popular opinions among hundreds of reviews, a significant
step towards opinion summarization of practical scope. In addition, QT enables
controllable summarization without further training, by utilizing properties of
the quantized space to extract aspect-specific summaries.


## Using our model

### Setting up the environment

* __Python version:__ `python3.6`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		pip install -U pip
		pip install -U setuptools
		pip install -r requirements.txt 

* __ROUGE:__ To ensure replicability and future research, we used the original
  ROUGE perl implementation and the `pyrouge` wrapper. Please follow the
instructions
[in this guide](https://github.com/stangelid/qt/blob/main/rouge-setup.md) to
setup ROUGE and `pyrouge` correctly. Make sure to you have activated your
conda/virtualenv environment when installing `pyrouge` 

* __SPACE training set__: The training set is not included in this repo. Download
SPACE via the above google drive link (405MB) and copy the file `space_train.json`
into the `./data/json/` directory.


### Training QT

To train QT on a subset of the training set using a GPU, go to the `./src`
directory and run the following:

    python3 train.py --max_num_entities 500 --run_id run1 --gpu 0

This will train a QT model with default hyperparameters (for general
summarization), store tensorboard logs under `./logs` and save a
model snapshot after every epoch under `./models` (filename:
`run1_<epoch>_model.pt`). Our model achieves high summarization performance,
even when trained on reviews from just 500 entities, as shown here.

For explanations of the available parameters for training the model, please see `train.py`.


### Summarization with QT

To perform general opinion summarization with a trained QT model, go to the `./src` directory and run the following:

	python3 extract.py --model ../models/run1_20_model.pt --sample_sentences --run_id general_run1 --gpu 0

This will store the summaries under `./outputs/general_run1` and also the output of ROUGE evaluation in `./outputs/eval_general_run1.json`. The `--sample_sentences` flag enables 2-step sampling.

For aspect opinion summarization, run:

	python3 aspect_extract.py --model ../models/run1_20_model.pt --sample_sentences --run_id aspects_run1 --gpu 0
	
Outputs stored similarly to the general opinion summarization example. For explanations of the available parameters for summarizing with the model, please see `extract.py` and `aspect_extract.py`.

### Hyperparameters used in paper

Check [this page](https://github.com/stangelid/qt/blob/main/hyperparams.md) for
details on the hyperparameters used in our paper's main experiments.

### Using QT on a custom corpus

If you want to use QT with a summarization corpus other than SPACE, please
follow [the instruction on this page](https://github.com/stangelid/qt/blob/main/custom.md).
