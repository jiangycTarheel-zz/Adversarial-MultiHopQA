# !/bin/bash
set -eu -o pipefail

mkdir -p data
cd data

### GloVe vectors ###
if [ ! -d glove ]
then
  mkdir glove
  cd glove
  wget http://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
  cd ..
fi

### HotpotQA ###
if [ ! -d hotpotqa ]
then
  mkdir hotpotqa
  cd hotpotqa
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
  cd ..
fi
cd ..

cd nectar
mkdir -p lib
cd lib

# CoreNLP 3.6.0
corenlp='stanford-corenlp-full-2015-12-09'
if [ ! -d "${corenlp}" ]
then
  wget "http://nlp.stanford.edu/software/${corenlp}.zip"
  unzip "${corenlp}.zip"
  ln -s "${corenlp}" stanford-corenlp
fi

cd ..
cd ..
pwd
mkdir resources
mkdir out
ln -s ../nectar/nectar resources
