# Adversarial-MultiHopQA
* Official code for our [ACL 2019 paper](https://arxiv.org/pdf/1906.07132.pdf).
* The initial code was adapted from [Adversarial Squad](https://github.com/robinjia/adversarial-squad).
* We adapt the Corenlp code (nectar) to support python 3.
* The arithmetic on Glove word embeddings are adapted from (https://github.com/brannondorsey/GloVe-experiments).

### Dependencies
Run `download_data.sh` to pull HotpotQA data and GloVe vectors.
* We tested our code on TF1.3, TF1.8, TF1.11 and TF1.13.
* See `requirements.txt`.

### 1. Preprocess the data using Corenlp
Run:
```
python3 convert_sp_facts.py corenlp -d dev
```
to store preprocessed data in `data/hotpotqa/dev_corenlp_cache_***.json`. This avoids rerunning Corenlp every time we generate an adversarial data.
If you want to create the adversarial training data, run:
```
python3 convert_sp_facts.py corenlp -d train
```
Warning: preprocessing both the training set and dev set requires a storage space of ~22G.


### 2. Collect the candidate answer and title set 
Run:
```
python3 convert_sp_facts.py gen-answer-set -d dev
```
and 
```
python3 convert_sp_facts.py gen-title-set -d dev
```
This step collect all answers and Wikipedia article titles in the dev set and classify them based on their NER and POS tag.


### 3. (Optional) Collect all paragraphs appearining in the context
If you want to eliminate the title-balancing bias in the adversarial documents (described in the last paragraph of Sec. 2.2), run: 
```
python3 convert_sp_facts.py gen-all-docs -d dev
```  

### 4. Generate Adverarial Dev set
To generate the adversarial dev set described in our paper, run:
```
python3 convert_sp_facts.py dump-addDoc -d dev -b --rule wordnet_dyn_gen --replace_partial_answer --num_new_doc=4 --dont_replace_full_answer --find_nearest_glove --add_doc_incl_adv_title
```
This will create the adversarial training set in `out/hotpot_dev_addDoc.json`
Note: `--add_doc_incl_adv_title` can be set only if Step 3 is done.


### 5. Generate Adverarial Training set
Generating the adversarial training set all at once could take days. Therefore, we divide the training set into 19 batches with the size of 5000, and process each batch in a separate program by running:
```
python3 convert_sp_facts.py dumpBatch-addDoc -d train -b --rule wordnet_dyn_gen --replace_partial_answer --num_new_doc=4 --dont_replace_full_answer --find_nearest_glove --add_doc_incl_adv_title --batch_idx=0
```
with `batch_idx` set to 0~18. After they finish, run:
```
python3 convert_sp_facts.py merge_files -d train
```
This will create the adversarial training set in `out/hotpot_train_addDoc.json`

#### Huggingface Datasets
A dataset created following the procedure above is hosted on [Huggingface datasets](https://huggingface.co/datasets/sagnikrayc/adversarial_hotpotqa). The original authors bear no responsibility for this HF dataset and the uploader there should be contacted in case of any discrepancies. 

**In order to recreate the adversarial training data we used in the paper, randomly sample 40% of the adversarial training data generated using this code and combine with the original HotpotQA training set.**


# Citation
```
@inproceedings{Jiang2019reasoningshortcut, 
	author={Yichen Jiang and Mohit Bansal}, 
	booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics}, 
	title={Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA}, 
	year={2019}, 
}
```
