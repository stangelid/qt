# Setting up ROUGE and pyrouge

## Installing ROUGE-1.5.5

Download ROUGE-1.5.5 from [here](https://github.com/andersjo/pyrouge). You only need to take note of the ROUGE-1.5.5 directory.

```bash
git clone https://github.com/andersjo/pyrouge.git
```

### Check if perl is installed

Run the command `perl` and, if it is not already installed on your system, follow the steps 2, 3 and 4 [here](https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/).

### Set the ROUGE environmental variable

An environment variable ROUGE_EVAL_HOME must be set to the absolute path to the `data` directory within the ROUGE-1.5.5 directory:

```bash
export ROUGE_EVAL_HOME="/absolute/path/to/ROUGE-1.1.5/data/"
```

You can put the above command in your `.bashrc file`, so you don't need to re-run it each time you open a new bash shell.

### Avoiding WordNet exceptions

To avoid any WordNet exceptions, run these commands:

```bash
cd /absolute/path/to/ROUGE-1.1.5/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db

cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

## Installing pyrouge

### Get the correct version

As of now, pypi version of pyrouge is deprecated, so let’s get the latest
version from the repository:

```bash
git clone https://github.com/bheinzerling/pyrouge.git
cd pyrouge
```

### Setup

```bash
sudo python setup.py install
pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/
```

### Test if everything’s right

```bash
cd pyrouge
python test.py
```
