## Installing ROUGE-1.5.5

Clone the (deprecated) pyrouge from the repo below -- you just need to get ROUGE-1.5.5 from it! Place the ROUGE-1.5.5 directory anywhere in your system, and take note of its absolute path for later.

```bash
git clone https://github.com/andersjo/pyrouge.git
cp -r pyrouge/tools/ROUGE-1.5.5 somewhere/in/your/filesystem/
realpath somewhere/in/your/filesystem/ROUGE-1.5.5  # this will give you the absolute path
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
rm -f WordNet-2.0.exc.db
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

## Installing pyrouge

### Get the correct version

As of now, pypi version of pyrouge is deprecated, so let’s get the latest
version from the repository:

```bash
git clone https://github.com/bheinzerling/pyrouge.git
```

### Setup

```bash
cd pyrouge
sudo python setup.py install
cd bin
pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/
```

### Test if everything’s right

```bash
cd ../pyrouge
python test.py
```
