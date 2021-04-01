# Using QT on a custom corpus

### Data preparation

First, you will need to prepare your dataset in the appropriate json
format. Here is how the training set should look like (no reference
summaries):

```
[
  {
    "entity_id": "...",
    "reviews": [
      {
        "review_id": "...",
        "rating": 3,
        "sentences": [
          "first sentence text",
          "second sentence text", 
          ...
        ]
      },
      ...
    ]
  },
  ...
]
```

Gold summarization data should go on a separate json and also include a
"split" (dev/test) and `"summaries"` fields for every entity. General summaries
should go under the `"general"` key, as shown:

    [
      {
        "entity_id": "...",
        "split": "dev",
        "reviews": [
          {
            "review_id": "...",
            "rating": 3,
            "sentences": [
              "first sentence text",
              "second sentence text", 
              ...
            ]
          },
          ...
        ],
        "summaries": {
          "general": [
            "reference summary 1 text",
            "reference summary 2 text",
            ...
          ],
          "aspect1": [...],
        }
      },
      ...
    ]

Note that the reviews need to be pre-split into sentences (for efficient
loading), but the summaries don't. You can check the `space_train.json` and
`space_summ.json` files for examples. From here on, let's assume custom training
and summarization data files `mydata_train.json` & `mydata_summ.json` under the
`./data/json/` directory.

Next, you need to write the reference summaries into separate files (to be used
by ROUGE). You can use the `json-to-dirs.py` script for this:

    cd ./src/utils/
    python3 json-to-dirs.py ../../data/mydata_summ.json ../../data/mygold

This will create aspect-specific subdirectories under `mygold` and write
summary files into them.

If you intend to perform aspect summariztion, you need to specify a
ranked list of _seed_ query words that describe every aspect. See examples
from SPACE
[here](https://github.com/stangelid/qt/blob/main/data/seeds/location.txt). The
filenames must be `<aspect>.txt` as in the provided files. Use a _dummy_ seed word
score of 1 as the first column for all seeds (we don't use the scores in the
current model). You can specify how many seed words to use when you run the
summarizer. Place all seed files under `./data/myseeds/` 

Finally, you need to train a SentencePiece tokenizer on your data using our
`train-spm.py` script

    cd ./src/utils/
    python3 train-spm.py path/to/mydata_train.json myspm
    mv myspm* ../../data/sentencepiece/

### Training QT

To train QT on your data with default hyperparameters, use the following:

    python3 train.py --data ../data/json/mydata_train.json --sentencepiece ../data/sentencepiece/myspm.model --run_id mydata_run1 --gpu 0

### Summarizing with QT

For general summarization (with 2-step sampling):

    python3 extract.py --summary_data ../data/json/mydata_summ.json --gold_data ../data/mygold --sentencepiece ../data/sentencepiece/myspm.model --split_by presplit --model ../models/mydata_run1_20_model.pt --sample_sentences --gpu 0 --run_id mydata_gen_run1

For aspect summarization:

    python3 aspect_extract.py --summary_data ../data/json/mydata_summ.json --gold_data ../data/mygold --sentencepiece ../data/sentencepiece/myspm.model --split_by presplit --model ../models/mydata_run1_20_model.pt --seedsdir ../data/myseeds --sample_sentences --gpu 0 --run_id mydata_asp_run1

