import sys
import collections
import json
import math
import os
import string
import re
import numpy as np
import itertools
import random
import nltk
import urllib.parse
from tqdm import tqdm
# sys.path.append('./en/wordnet')
# sys.path.append('./en')
sys.path.append('./resources/nectar/corenlp')
sys.path.append('./resources/nectar/base')
sys.path.append('./resources/nectar')
import argparse
from resources.nectar import corenlp
from answer_rules import ans_date, ans_number, ans_entity_full, ans_abbrev, ans_match_wh, ans_pos, ans_catch_all, ans_wordnet_catch_amap, lookup_answer_generate
from bridge_entity_rules import bridge_entity_full, bridge_abbrev, bridge_wordnet_catch_amap, bridge_pos, bridge_number, lookup_title_generate, bridge_date
from nearest_glove.get_nearest import build_glove_matrix, find_nearest_word
from nltk.corpus import stopwords
COMMON_WORDS = list(stopwords.words('english'))
PUNCTUATIONS = list(string.punctuation) + ['–']
SHORT_PUNCT = PUNCTUATIONS.copy() + [' ']
PUNCT_COMMON = PUNCTUATIONS + COMMON_WORDS + [' ']
del SHORT_PUNCT[SHORT_PUNCT.index('+')]
OPTS = None
SOURCE_DIR = os.path.join("data", "hotpotqa")
DATASETS = {
  'dev': os.path.join(SOURCE_DIR, 'hotpot_dev_distractor_v1.json'),
  'train': os.path.join(SOURCE_DIR, 'hotpot_train_v1.1.json'),
}

CORENLP_CACHES = {
  'dev': 'data/hotpotqa/dev_corenlp_cache_',
  'train': 'data/hotpotqa/train_corenlp_cache_',
}

COMMANDS = ['corenlp', 'dump-addDoc', 'gen-answer-set', 'gen-title-set', 'gen-all-docs']
CORENLP_PORT = 8765
CORENLP_LOG = 'corenlp.log'
CATCH_ALL_NUM = 0

def parse_args():
  parser = argparse.ArgumentParser('Generate adversarial support facts for HotpotQA.')
  parser.add_argument('command',
                      help='Command (options: [%s]).' % (', '.join(COMMANDS))) 
  parser.add_argument('--substitute_bridge_entities', '-b', default=False, action='store_true')
  parser.add_argument('--dataset', '-d', default='dev',
                      help='Which dataset (options: [%s])' % (', '.join(DATASETS)))
  parser.add_argument('--prepend', '-p', default=False, 
                      action='store_true', help='Prepend fake answer to the original answer.')
  parser.add_argument('--quiet', '-q', default=False, action='store_true')
  parser.add_argument('--rule', '-r', default='wordnet_dyn_gen', help='[wordnet | wordnet_dyn_gen]')
  parser.add_argument('--seed', '-s', default=-1, type=int, help='Shuffle with RNG seed.')
  parser.add_argument('--split', default='0-1000')
  parser.add_argument('--replace_partial_answer', default=False, action='store_true')
  parser.add_argument('--num_new_doc', default=1, type=int)
  parser.add_argument('--dont_replace_full_answer', default=False, action='store_true', help='If true, only a few answer words, instead of the full answer span, will be replaced')
  parser.add_argument('--dont_replace_full_title', default=False, action='store_true', help='If true, only a few title words, instead of the full title span, will be replaced')
  parser.add_argument('--find_nearest_glove', default=False, action='store_true')
  parser.add_argument('--num_glove_words_to_use', default=100000, type=int)
  parser.add_argument('--batch_idx', default=0, type=int)
  parser.add_argument('--batch_size', default=5000, type=int)
  parser.add_argument('--add_doc_incl_adv_title', default=False, action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def read_data():
  filename = DATASETS[OPTS.dataset]
  with open(filename) as f:
    return json.load(f)


def run_corenlp(dataset, bsz=5000):
  with corenlp.CoreNLPServer(port=CORENLP_PORT, logfile=CORENLP_LOG) as server:
    client = corenlp.CoreNLPClient(port=CORENLP_PORT)
    for ib in range(int(len(dataset)/bsz) + 1):
      cache_file = CORENLP_CACHES[OPTS.dataset] + str(ib*bsz) + '-' + str(min((ib+1)*bsz, len(dataset))) + '.json'
      print(cache_file)
      cache = {}
      print('Running NER for paragraphs...')
      for ie, e in tqdm(enumerate(dataset[ib*bsz : (ib+1)*bsz])):
        context, question, answer, supports = e['context'], e['question'], e['answer'], e['supporting_facts']
        titles, partial_titles, sp_doc_ids = [], [], []
        for si, doc in enumerate(context):
          title = doc[0]
          titles.append(title)
          title_split = re.split("([{}])".format("()"), title)
          if title_split[0] != '':
            partial_titles.append(title_split[0])
          elif title_split[-1] != '':
            partial_titles.append(title_split[-1])
          else:
            real_title = title_split[title_split.index('(')+1]
            assert real_title != ')'
            partial_titles.append(real_title)
        for sp_doc_title, sent_id in supports:
          sp_doc_id = titles.index(sp_doc_title)
          if sp_doc_id not in sp_doc_ids:
            sp_doc_ids.append(sp_doc_id)

        response_context, response_title = [], []
        for _id, doc in enumerate(context):
          response_doc = []
          response_context.append(response_doc)
          response_title.append(client.query_ner(partial_titles[_id]))
          if _id not in sp_doc_ids:
            continue
          for sent in doc[1]:
            if sent == '' or sent == ' ':
              continue
            response = client.query_ner(sent)
            response_doc.append(response)
          
        cache[e['_id']] = [response_context]
        response_a = client.query_ner(answer)
        cache[e['_id']].append(response_a)
        response_q = client.query_ner(question)
        cache[e['_id']].append(response_q)
        cache[e['_id']].append(response_title)

      print('Dumping caches...')
      with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def load_cache(start, end):
  cache_file = CORENLP_CACHES[OPTS.dataset] + str(start) + '-' + str(end) + '.json'
  print(cache_file)
  with open(cache_file, 'r') as f:
    return json.load(f)


def load_ans_title_cache(source):
  cache_file = 'data/' + OPTS.dataset + '_' + source + '_set.json'
  print(cache_file)
  with open(cache_file, 'r') as f:
    return json.load(f)


def load_inv_index():
  filepath = 'data/inv_index.json'
  print("reading inv_index.json")
  with open(filepath) as f:
    return json.load(f)


FIXED_VOCAB_ANSWER_RULES = [
    ('date', ans_date),
    ('number', ans_number),
    ('ner_person', ans_entity_full('PERSON', 'Jeff Dean')),
    ('ner_location', ans_entity_full('LOCATION', 'Chicago')),
    ('ner_organization', ans_entity_full('ORGANIZATION', 'Stark Industries')),
    ('ner_misc', ans_entity_full('MISC', 'Jupiter')),
    ('abbrev', ans_abbrev('LSTM')),
    ('wh_who', ans_match_wh('who', 'Jeff Dean')),
    ('wh_when', ans_match_wh('when', '1956')),
    ('wh_where', ans_match_wh('where', 'Chicago')),
    ('wh_how_many', ans_match_wh('how many', '42')),
    # Starts with verb
    ('pos_begin_vb', ans_pos('VB', 'learn')),
    ('pos_end_vbd', ans_pos('VBD', 'learned')),
    ('pos_end_vbd', ans_pos('VBN', 'learned')),
    ('pos_end_vbg', ans_pos('VBG', 'learning')),
    ('pos_end_vbp', ans_pos('VBP', 'learns')),
    ('pos_end_vbz', ans_pos('VBZ', 'learns')),
    # Ends with some POS tag
    ('pos_end_nn', ans_pos('NN', 'hamster', end=True, add_dt=True)),
    ('pos_end_nnp', ans_pos('NNP', 'Central Park', end=True, add_dt=True)),
    ('pos_end_nns', ans_pos('NNS', 'hamsters', end=True, add_dt=True)),
    ('pos_end_nnps', ans_pos('NNPS', 'Kew Gardens', end=True, add_dt=True)),
    ('pos_end_jj', ans_pos('JJ', 'deep', end=True)),
    ('pos_end_jjr', ans_pos('JJR', 'deeper', end=True)),
    ('pos_end_jjs', ans_pos('JJS', 'deepest', end=True)),
    ('pos_end_rb', ans_pos('RB', 'silently', end=True)),
    ('pos_end_vbg', ans_pos('VBG', 'learning', end=True)),
    ('catch_all', ans_catch_all('aliens')),
]


WORDNET_ANSWER_RULES_DYN_GEN = [
    ('date', ans_date),
    ('number', ans_number),
    ('ner_person', lookup_answer_generate(ans_entity_full('PERSON', 'Jeff Dean'),
        'ner_person')),
    ('ner_location', lookup_answer_generate(ans_entity_full('LOCATION', 'Chicago'),
        'ner_location')),
    ('ner_organization', lookup_answer_generate(ans_entity_full('ORGANIZATION',
    'Stark Industries'), 'ner_organization')),
    ('wordnet_catch', ans_wordnet_catch_amap),
    ('ner_misc', lookup_answer_generate(ans_entity_full('MISC', 'Jupiter'),
        'ner_misc')),
    ('abbrev', lookup_answer_generate(ans_abbrev('LSTM'), 'abbrev')),
    ('wh_who', lookup_answer_generate(ans_match_wh('who', 'Jeff Dean'), 'wh_who')),
    ('wh_when', lookup_answer_generate(ans_match_wh('when', '1956'), 'wh_when')),
    ('wh_where', lookup_answer_generate(ans_match_wh('where', 'Chicago'), 'wh_where')),
    ('wh_how_many', lookup_answer_generate(ans_match_wh('how many', '42'),
        'wh_how_many')),
    # Starts with verb
    ('pos_begin_vb', lookup_answer_generate(ans_pos('VB', 'learn'), 'pos_begin_vb')),
    ('pos_end_vbd', lookup_answer_generate(ans_pos('VBD', 'learned'), 'pos_end_vbd')),
    ('pos_end_vbg', lookup_answer_generate(ans_pos('VBG', 'learning'), 'pos_end_vbg')),
    ('pos_end_vbp', lookup_answer_generate(ans_pos('VBP', 'learns'), 'pos_end_vbp')),
    ('pos_end_vbz', lookup_answer_generate(ans_pos('VBZ', 'learns'), 'pos_end_vbz')),
    # Ends with some POS tag
    ('pos_end_nn', lookup_answer_generate(ans_pos('NN', 'hamster', end=True,
        add_dt=True), 'pos_end_nn')),
    ('pos_end_nnp', lookup_answer_generate(ans_pos('NNP', 'Central Park', end=True,
        add_dt=True), 'pos_end_nnp')),
    ('pos_end_nns', lookup_answer_generate(ans_pos('NNS', 'hamsters', end=True,
        add_dt=True), 'pos_end_nns')),
    ('pos_end_nnps', lookup_answer_generate(ans_pos('NNPS', 'Kew Gardens', end=True, add_dt=True),
        'pos_end_nnps')),
    ('pos_end_jj', lookup_answer_generate(ans_pos('JJ', 'deep', end=True),
        'pos_end_jj')),
    ('pos_end_jjr', lookup_answer_generate(ans_pos('JJR', 'deeper', end=True),
        'pos_end_jjr')),
    ('pos_end_jjs', lookup_answer_generate(ans_pos('JJS', 'deepest', end=True),
        'pos_end_jjs')),
    ('pos_end_rb', lookup_answer_generate(ans_pos('RB', 'silently', end=True),
        'pos_end_rb')),
    ('pos_end_vbg', lookup_answer_generate(ans_pos('VBG', 'learning', end=True),
        'pos_end_vbg')),
    ('catch_all', lookup_answer_generate(ans_catch_all('aliens'),
        'catch_all'))
]


def get_tokens_from_coreobj(original, obj):
  """Get CoreNLP tokens corresponding to a SQuAD answer object."""
  toks = []
  for s in obj['sentences']:
    for t in s['tokens']:
      toks.append(t)
  if corenlp.rejoin(toks).strip() == original.strip():
    # Make sure that the tokens reconstruct the answer
    return toks
  else:
    if len(toks) == 0 and original == ' 🇦🇷':
      return toks
    assert False, (toks, corenlp.rejoin(toks).strip(), original)


def get_determiner_for_answer(answer):
  words = answer.split(' ')
  if words[0].lower() == 'the': return 'the'
  if words[0].lower() in ('a', 'an'): return 'a'
  return None

  
def process_sp_facts(e, answer_tok):
  """
  Return:
  sp_fact_w_answer: A list of at most 2 sublists, each sublist is a list of [Integer: doc_id, List: [Integer: sent_id]].
                    Representing the sentence-level supporting facts that contain the answer.
  sp_doc_w_answer: A list of at most 2 tuple (Integer: doc_id, String: doc_words).
                   Representing the document-level supporting facts that contain the answer.
  sp_doc_wo_answer: same as above. Representing the document-level supporting facts containing no answer.
  sp_doc_ids: A list of two integers.
  """
  supports, context, answer = e['supporting_facts'], e['context'], e['answer']
  titles, _context = [], []
  for si, doc in enumerate(context):
    _context.append(doc[1])
    titles.append(doc[0])
  sp_doc_w_answer_ids, sp_doc_wo_answer_ids, sp_doc_ids = [], [], []
  sp_fact_w_answer, sp_doc_wo_answer, sp_doc_w_answer = [], [], []

  sp_doc_sent_id, sp_doc_ids = [], []
  for sp_doc_title, sent_id in supports:
    sp_doc_id = titles.index(sp_doc_title)
    if sp_doc_id not in sp_doc_ids:
      sp_doc_ids.append(sp_doc_id) 
      sp_doc_sent_id.append([sp_doc_id, [sent_id]])
    else:
      sp_doc_sent_id[-1][1].append(sent_id)

  for sp_doc_id, sent_ids in sp_doc_sent_id:
    sent_ids = range(len(context[sp_doc_id][1]))

    for sent_id in sent_ids:
      if sent_id == 902:
        continue
      sent = _context[sp_doc_id][sent_id]
      a = re.search(r'({})'.format(re.escape(answer.lower())), sent.lower())
      if a:
        if sp_doc_id not in sp_doc_w_answer_ids:
          sp_fact_w_answer.append([sp_doc_id, [sent_id]])
          sp_doc_w_answer_ids.append(sp_doc_id)
        else:
          assert sp_doc_id == sp_fact_w_answer[-1][0]
          sp_fact_w_answer[-1][1].append(sent_id)        
      else:
        found_tok, not_found_tok = 0, 0
        for at in answer_tok:
          if at['originalText'] in PUNCT_COMMON:
            continue
          a = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(at['originalText'].lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), sent.lower())
          if a:
            found_tok += 1
          else:
            not_found_tok += 1
        if found_tok >= not_found_tok:
          if sp_doc_id not in sp_doc_w_answer_ids:
            sp_fact_w_answer.append([sp_doc_id, [sent_id]])
            sp_doc_w_answer_ids.append(sp_doc_id)
          else:
            assert sp_doc_id == sp_fact_w_answer[-1][0]
            sp_fact_w_answer[-1][1].append(sent_id)      

  for id in sp_doc_ids:
    if id in sp_doc_w_answer_ids:
      sp_doc_w_answer.append((id,''.join(_context[id])))
    else:
      sp_doc_wo_answer.append((id,''.join(_context[id])))

  return sp_fact_w_answer, sp_doc_w_answer, sp_doc_wo_answer, sp_doc_ids


FIXED_VOCAB_BRIDGE_RULES = [
    ('ner_person', bridge_entity_full('PERSON', 'Jeff Dean')),
    ('ner_location', bridge_entity_full('LOCATION', 'Chicago')),
    ('ner_organization', bridge_entity_full('ORGANIZATION', 'Stark Industries')),
    ('ner_misc', bridge_entity_full('MISC', 'Jupiter')),
    ('abbrev', bridge_abbrev('LSTM')),
    # Starts with verb
    ('pos_begin_vb', bridge_pos('VB', 'act')),
    ('pos_end_vbd', bridge_pos('VBD', 'acted')),
    ('pos_end_vbd', bridge_pos('VBN', 'acted')),
    ('pos_end_vbg', bridge_pos('VBG', 'acting')),
    ('pos_end_vbp', bridge_pos('VBP', 'acts')),
    ('pos_end_vbz', bridge_pos('VBZ', 'acts')),
    # Ends with some POS tag
    ('pos_end_nn', bridge_pos('NN', 'table', end=True)),
    ('pos_end_nnp', bridge_pos('NNP', 'Hyde Park', end=True)),
    ('pos_end_nns', bridge_pos('NNS', 'tables', end=True)),
    ('pos_end_nnps', bridge_pos('NNPS', 'Trump Towers', end=True)),
    ('pos_end_jj', bridge_pos('JJ', 'hard', end=True)),
    ('pos_end_jjr', bridge_pos('JJR', 'harder', end=True)),
    ('pos_end_jjs', bridge_pos('JJS', 'hardest', end=True)),
    ('pos_end_rb', bridge_pos('RB', 'loudly', end=True)),
    ('date', ans_date),
    ('number', bridge_number),
    ('catch_all', ans_catch_all('players')),
]

BRIDGE_RULES_DYN_GEN = [
    ('date', bridge_date),
    ('number', bridge_number),
    ('ner_person', lookup_title_generate(bridge_entity_full('PERSON', 'Jeff Dean'), 'ner_person')),
    ('ner_location', lookup_title_generate(bridge_entity_full('LOCATION', 'Chicago'), 'ner_location')),
    ('ner_organization', lookup_title_generate(bridge_entity_full('ORGANIZATION', 'Stark Industries'), 'ner_organization')),
    ('ner_misc', lookup_title_generate(bridge_entity_full('MISC', 'Jupiter'), 'ner_misc')),
    ('abbrev', lookup_title_generate(bridge_abbrev('LSTM'), 'abbrev')), 
    # Starts with verb
    ('pos_begin_vb', lookup_title_generate(bridge_pos('VB', 'learn'), 'pos_begin_vb')),
    ('pos_end_vbd', lookup_title_generate(bridge_pos('VBD', 'learned'), 'pos_end_vbd')),
    ('pos_end_vbg', lookup_title_generate(bridge_pos('VBG', 'learning'), 'pos_end_vbg')),
    ('pos_end_vbp', lookup_title_generate(bridge_pos('VBP', 'learns'), 'pos_end_vbp')),
    ('pos_end_vbz', lookup_title_generate(bridge_pos('VBZ', 'learns'), 'pos_end_vbz')),
    # Ends with some POS tag
    ('pos_end_nn', lookup_title_generate(bridge_pos('NN', 'hamster', end=True), 'pos_end_nn')),
    ('pos_end_nnp', lookup_title_generate(bridge_pos('NNP', 'Central Park', end=True), 'pos_end_nnp')),
    ('pos_end_nns', lookup_title_generate(bridge_pos('NNS', 'hamsters', end=True), 'pos_end_nns')),
    ('pos_end_nnps', lookup_title_generate(bridge_pos('NNPS', 'Kew Gardens', end=True), 'pos_end_nnps')),
    ('pos_end_jj', lookup_title_generate(bridge_pos('JJ', 'deep', end=True), 'pos_end_jj')),
    ('pos_end_jjr', lookup_title_generate(bridge_pos('JJR', 'deeper', end=True), 'pos_end_jjr')),
    ('pos_end_jjs', lookup_title_generate(bridge_pos('JJS', 'deepest', end=True), 'pos_end_jjs')),
    ('pos_end_rb', lookup_title_generate(bridge_pos('RB', 'silently', end=True), 'pos_end_rb')),
    ('pos_end_vbg', lookup_title_generate(bridge_pos('VBG', 'learning', end=True), 'pos_end_vbg')),
    ('catch_all', lookup_title_generate(ans_catch_all('aliens'), 'catch_all')),
]


def create_fake_answer(answer, a_toks, question, determiner, ans_cache=None):
  if OPTS.rule == 'fixed_vocab':
    rules = FIXED_VOCAB_ANSWER_RULES
  elif OPTS.rule == 'wordnet_dyn_gen':
    rules = WORDNET_ANSWER_RULES_DYN_GEN
  else:
    raise NotImplementedError
  for rule_name, func in rules:
    new_answer = func(answer, a_toks, question, ans_cache=ans_cache, determiner=determiner)
    if new_answer and (rule_name == 'date' or rule_name == 'number'):
      return new_answer, None
    if new_answer: break
  else:
    raise ValueError('Missing answer')
  return new_answer


def create_fake_bridge_entity(entity, t_toks, q, title_cache=None, is_end=False):
  if OPTS.rule == 'fixed_vocab':
    rules = FIXED_VOCAB_BRIDGE_RULES
  elif OPTS.rule == 'wordnet_dyn_gen':
    rules = BRIDGE_RULES_DYN_GEN
  else:
    raise NotImplementedError
  for rule_name, func in rules:
    new_entity = func(entity, t_toks, q, title_cache=title_cache, is_end=is_end)
    if new_entity: break
  else:
    raise ValueError('Missing entity')
  return new_entity


def find_bridge_entities(tok_doc1, tok_doc2, tok_question, tok_answer, tok_title, find_title_only=False):
  """
  Args:
  tok_doc1: [num_sent, sent_len]
  tok_doc2: [doc_len]
  """
  doc1_wordss = [[t['originalText'].lower() for t in s] for s in tok_doc1]
  doc1_ners = [[t['ner'] for t in s] for s in tok_doc1]
  doc2_words = [t['originalText'].lower() for t in tok_doc2]
  doc2_ner = [t['ner'] for t in tok_doc2]
  question_words = [t['originalText'].lower() for t in tok_question]
  answer_words = [t['originalText'].lower() for t in tok_answer]
  title_words = [t['originalText'].lower() for t in tok_title]
  title_ner = [t['ner'] for t in tok_title]
  entity_list, entity_idx = [], {}
  ngram_entity_list, ngram_entity_idx = [], {}
  ngram_entity = []

  for i in range(2):
    if i == 0:
      words_to_look, ners_to_look = title_words, title_ner
    else:
      words_to_look, ners_to_look = doc2_words, doc2_ner

    for iss, (doc1_words, doc1_ner) in enumerate(zip(doc1_wordss, doc1_ners)):
      end_entity = True
      for ie, (entity, ner) in enumerate(zip(doc1_words[::-1], doc1_ner[::-1])):
        if entity in words_to_look and entity not in answer_words and entity not in question_words and entity not in PUNCTUATIONS+COMMON_WORDS:
          if i == 1 and (ner == 'O' and ners_to_look[words_to_look.index(entity)] == 'O'):
            continue
          if entity not in entity_list:
            entity_list.append(entity)
            entity_idx[entity] = [end_entity, [(iss, len(doc1_words)-1-ie)]]
          else:
            entity_idx[entity][0] = entity_idx[entity][0] | end_entity
            entity_idx[entity][1].append([iss, len(doc1_words)-1-ie])
          ngram_entity.append(entity)
          end_entity = False
        else:
          if len(ngram_entity) > 1:
            ngram_entity_list.append(ngram_entity[::-1])
          ngram_entity = []
          end_entity = True
    if find_title_only:
      break

  return entity_list, entity_idx


def create_adversarial_exammple(data, adv_strategy='addDoc', start=0, end=5000, ans_cache=None, title_cache=None, glove_tools=None,
  all_docs=None):
  corenlp_cache = load_cache(start, end)
  if OPTS.find_nearest_glove:
    assert glove_tools
    df, word_to_id, id_to_word = glove_tools[0], glove_tools[1], glove_tools[2]

  unmatched_qas = []
  num_matched = 0
  new_sp_facts = {}

  for ie, e in tqdm(enumerate(data[start : end])):
    answer = e['answer']
    if answer == 'yes' or answer == 'no':
      continue
    context, question = e['context'], e['question']
    context_parse, answer_parse, question_parse, title_parse = corenlp_cache[e['_id']]
    answer_tok = get_tokens_from_coreobj(answer, answer_parse)
    determiner = get_determiner_for_answer(answer)

    sps_w_answer, sp_docs_w_answer, sp_docs_wo_answer, sp_doc_ids = process_sp_facts(e, answer_tok)
    assert len(sp_doc_ids) == 2
    non_sp_doc_ids = [_i for _i in range(len(e['context'])) if _i not in sp_doc_ids]
    
    assert len(sps_w_answer) == len(sp_docs_w_answer), (ie, sps_w_answer)
    new_docs, new_docs_2 = [], []

    titles = []
    for si, doc in enumerate(context):
      title = doc[0]
      title_split = re.split("([{}])".format("()"), title)
      if title_split[0] != '':
        titles.append(title_split[0])
      elif title_split[-1] != '':
        titles.append(title_split[-1])
      else:
        real_title = title_split[title_split.index('(')+1]
        assert real_title != ')'
        titles.append(real_title)

    if OPTS.find_nearest_glove:
      new_entities_dict = {}
    
    for isp, (sp_doc_id, sps_in_one_doc) in enumerate(sps_w_answer):
      # All sp_facts in a single doc should get the same fake answer.
      new_ans, new_ans_tok = create_fake_answer(answer, answer_tok, question, determiner=determiner, ans_cache=ans_cache)  
      ans_len_diff = len(new_ans) - len(answer)
      if new_ans and isp == 0:
        num_matched += 1
      if new_ans is None:
        unmatched_qas.append((question, answer))
      
      for _ind in range(OPTS.num_new_doc):
        if _ind > 0:
          new_ans, new_ans_tok = create_fake_answer(answer, answer_tok, question, determiner=determiner, ans_cache=ans_cache)  
          ans_len_diff = len(new_ans) - len(answer)
        new_doc = context[sp_doc_id][1].copy()

        ## Step 1: substitute any bridge entities
        if OPTS.substitute_bridge_entities and e['type'] == 'bridge':
          if len(sps_w_answer) == 1:
            doc_id_to_compare, doc_to_compare = sp_docs_wo_answer[0]
          else:  # There is no bridge entity in this case actually, only need to change the title entities.
            doc_id_to_compare, doc_to_compare = sp_docs_w_answer[(isp+1)%2]           
          assert doc_id_to_compare != sp_doc_id, (doc_id_to_compare, sp_doc_id)
          doc_tok_to_compare_list = [get_tokens_from_coreobj(s, s_parse) for (s, s_parse) in zip(context[doc_id_to_compare][1], context_parse[doc_id_to_compare])]
          doc_tok_to_compare = list(itertools.chain.from_iterable(doc_tok_to_compare_list))
          question_tok = get_tokens_from_coreobj(question, question_parse)

          new_doc_tok = [get_tokens_from_coreobj(_sent, context_parse[sp_doc_id][iis]) for iis, _sent in enumerate(new_doc) if _sent != ' ' and _sent != '']
          old_title = titles[sp_doc_id]
          old_title_compare = titles[doc_id_to_compare]
          title_tok = get_tokens_from_coreobj(old_title, title_parse[sp_doc_id])
          title_tok_to_compare = get_tokens_from_coreobj(titles[doc_id_to_compare], title_parse[doc_id_to_compare])
          for i in range(2):  # Substitute the title of both sp docs.
            if i == 1 and len(sps_w_answer) == 2:
              break
            if i == 0: 
              _title, _title_tok = old_title, title_tok
            else:
              _title, _title_tok = old_title_compare, title_tok_to_compare

            # First, substitute the title.
            new_title, new_title_tok = create_fake_bridge_entity(_title, _title_tok, question, title_cache=title_cache)
            if OPTS.add_doc_incl_adv_title and len(sps_w_answer) == 1 and i == 0:
              for _doc in all_docs:
                _doc_title = _doc[0]
                if '(' in new_title:
                  _new_title = new_title[:new_title.index('(')].strip()
                else:
                  _new_title = new_title
                if _doc_title != new_title:
                  if _new_title.lower() in ''.join(_doc[1]).lower():
                    new_docs_2.append(_doc)
                    break

            ent_len_diff = len(new_title) - len(_title)

            # Substitute the entire title.
            if new_title_tok is None or OPTS.dont_replace_full_title is False or len(_title_tok) == 1:
              foundd = 0
              for isent, _sent in enumerate(new_doc):
                try:
                  a = re.finditer(r'({})'.format('(?<!([A-Za-z]))'+re.escape(_title.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), _sent.lower())
                except:
                  print(_sent)
                  print(_title)
                  print(ie)
                  exit()
                for ifo, found in enumerate(a):
                  foundd += 1
                  assert len(_sent) > found.start()+ifo*ent_len_diff
                  _sent = _sent[:(found.start()+ifo*ent_len_diff)] + new_title + _sent[(found.end()+ifo*ent_len_diff):]
                new_doc[isent] = _sent
              if i == 0 and new_title_tok is None and foundd == 0:
                print(ie)
                assert False
            
            title_tok_wo_common = [t for t in _title_tok if t['originalText'] not in COMMON_WORDS + PUNCTUATIONS]
            num_title_tok_to_replace = 0
            if OPTS.dont_replace_full_title and len(title_tok_wo_common) > 1:
              title_tok_to_replace = []
              for _token in title_tok_wo_common:
                a = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(_token['originalText'].lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), question.lower())
                if a is None:
                  title_tok_to_replace.append(_token)
                  num_title_tok_to_replace += 1
              if len(title_tok_to_replace) == 0 or num_title_tok_to_replace == 0: # All tokens are mentioned in the question
                title_tok_to_replace = title_tok_wo_common

            else:
              title_tok_to_replace = title_tok_wo_common

            if new_title_tok:
              new_title_tok_wo_common = [t for t in new_title_tok if t['originalText'] not in COMMON_WORDS + PUNCTUATIONS]

              for ittok, ttok in enumerate(title_tok_to_replace):
                entity = ttok['originalText']
                if entity in PUNCTUATIONS + COMMON_WORDS + ['NONE']:
                  continue
                if ittok < len(new_title_tok_wo_common):
                  new_entity = new_title_tok_wo_common[ittok]['originalText']
                else:
                  new_entity = new_title_tok_wo_common[-1]['originalText']

                if new_entity in PUNCT_COMMON:
                  if new_title_tok_wo_common[-1]['originalText'] not in PUNCT_COMMON:
                    new_entity = new_title_tok_wo_common[-1]['originalText']
                  elif new_title_tok_wo_common[0]['originalText'] not in PUNCT_COMMON:
                    new_entity = new_title_tok_wo_common[0]['originalText']
                  else:
                    for _tok in new_title_tok_wo_common[1:-1]:
                      if _tok['originalText'] not in PUNCT_COMMON:
                        new_entity = _tok['originalText']
                assert new_entity not in PUNCT_COMMON

                ent_len_diff = len(new_entity) - len(entity)
                for isent, _sent in enumerate(new_doc):
                  a = re.finditer(r'({})'.format('(?<!([A-Za-z]))'+re.escape(entity.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), _sent.lower())
                  for ifo, found in enumerate(a):
                    assert len(_sent) > found.start()+ifo*ent_len_diff
                    _sent = _sent[:(found.start()+ifo*ent_len_diff)] + new_entity + _sent[(found.end()+ifo*ent_len_diff):]
                  new_doc[isent] = _sent

          new_docs.append(new_doc)
              
        for ispsent, sp_sent_id in enumerate(sps_in_one_doc):
          ## Step 2: replace the original answer in sp with new_ans.
          sp = new_doc[sp_sent_id]
          new_sp = sp

          if new_ans_tok is None or OPTS.dont_replace_full_answer is False or len(answer_tok) == 1:
            a = re.finditer(r'({})'.format(re.escape(answer.lower())), new_sp.lower())
            for ifo, found in enumerate(a):
              new_sp = new_sp[:(found.start()+ifo*ans_len_diff)] + new_ans + new_sp[(found.end()+ifo*ans_len_diff):]
          
          ## Step 3: add the adversarial sp 
          new_doc[sp_sent_id] = new_sp
            
        ## Step 4: replace partial answer
        if OPTS.replace_partial_answer and len(answer_tok) > 1 and new_ans_tok:
          # If the answer already doesn't exi
          answer_tok_wo_common = [t for t in answer_tok if t['originalText'] not in COMMON_WORDS + PUNCTUATIONS]
          _replace = False
          num_replaced = 0
          answer_tok_to_replace = []
          for _token in answer_tok_wo_common:
            a = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(_token['originalText'].lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), question.lower())
            if a is not None:
              answer_tok_to_replace.append(_token)

          if len(answer_tok_to_replace) == 0:
            answer_tok_to_replace = answer_tok_wo_common

          for ittok, ttok in enumerate(answer_tok_to_replace):
            entity = ttok['originalText']
            if entity in PUNCTUATIONS + COMMON_WORDS:
              continue
            if OPTS.dont_replace_full_answer and len(answer_tok_to_replace) > 1:
              if _replace is False:
                _replace = True
                continue
              else:
                _replace = False
            num_replaced += 1
            if OPTS.find_nearest_glove:
              if isp == 0 and _ind == 0:
                new_entities = find_nearest_word(entity.lower(), df, word_to_id, id_to_word)
                new_entities_dict[entity] = new_entities
              else:
                new_entities = new_entities_dict[entity]
            else:
              new_entities = None

            if new_entities and len(new_entities) > 0:
              new_entity = new_entities[0]
              new_entities_dict[entity] = new_entities[1:]
            else: 
              if ittok < len(new_ans_tok):
                new_entity = new_ans_tok[ittok]['originalText']
              else:
                new_entity = new_ans_tok[-1]['originalText']
              if new_entity in PUNCT_COMMON:
                if new_ans_tok[-1]['originalText'] not in PUNCT_COMMON:
                  new_entity = new_ans_tok[-1]['originalText']
                elif new_ans_tok[0]['originalText'] not in PUNCT_COMMON:
                  new_entity = new_ans_tok[0]['originalText']
                else:
                  for _tok in new_ans_tok[1:-1]:
                    if _tok['originalText'] not in PUNCT_COMMON:
                      new_entity = _tok['originalText']
              assert new_entity not in PUNCT_COMMON, (new_ans_tok)

            ent_len_diff = len(new_entity) - len(entity)
            for isent, _sent in enumerate(new_doc):
              a = re.finditer(r'({})'.format('(?<!([A-Za-z]))'+re.escape(entity.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), _sent.lower())
              for ifo, found in enumerate(a):
                assert len(_sent) > found.start()+ifo*ent_len_diff
                _sent = _sent[:(found.start()+ifo*ent_len_diff)] + new_entity + _sent[(found.end()+ifo*ent_len_diff):]
              new_doc[isent] = _sent

        if new_ans_tok and OPTS.dont_replace_full_answer and len(answer_tok) > 1: # In this case, the replace_full_answer procedure is skipped.
          if answer != '""':
            assert num_replaced > 0, (answer, new_ans, ie)

        a = re.search(r'({})'.format(re.escape('(?<!([A-Za-z]))'+ answer.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), ''.join(new_doc).lower())
        assert a is None, (new_doc, answer, ie) # Make sure that there is no answer in new doc.

    if OPTS.num_new_doc == 1:
      for new_doc in new_docs:
        if OPTS.prepend:
          insert_pos = 0
        else:
          insert_pos = np.random.randint(low=0, high=len(context) + 1)
        context.insert(insert_pos, ['added_{}'.format(str(isp)), new_doc])
    else: # Totally reform the context
      new_context = []
      for new_doc in new_docs:
        new_context.append(['added', new_doc])

      for sp_did in sp_doc_ids:
        new_context.append(context[sp_did])

      if OPTS.add_doc_incl_adv_title and len(new_context) <= 10:
        assert len(new_docs_2) <= 4
        for new_doc in new_docs_2:
          new_context.append(new_doc)

      if len(new_context) <= 10:
        for _rm in range(10 - len(new_context)):
          if _rm >= len(non_sp_doc_ids):
            break
          new_context.append(context[non_sp_doc_ids[_rm]])

      if OPTS.prepend is False:
        random.shuffle(new_context)
      
      e['context'] = new_context
  return data, num_matched


def dump_data(data, adv_strategy='addDoc', bsz=5000):
  if OPTS.add_doc_incl_adv_title:
    filepath = 'data/all_' + OPTS.dataset + '_docs.json'
    all_docs = json.load(open(filepath))
  else:
    all_docs = None

  total_num_matched = 0
  ans_cache, title_cache = None, None
  if adv_strategy == 'addDoc' and 'dyn_gen' in OPTS.rule:
    ans_cache = load_ans_title_cache('answer')
    title_cache = load_ans_title_cache('title')

  glove_tools = None
  if OPTS.find_nearest_glove:
    assert OPTS.dont_replace_full_answer
    glove_tools = build_glove_matrix(num_words=OPTS.num_glove_words_to_use)

  all_data = []
  for i in range(int(len(data)/bsz) + 1):
    data, num_matched = create_adversarial_exammple(data, adv_strategy, start=i*bsz, end=min((i+1)*bsz, len(data)), 
        ans_cache=ans_cache, title_cache=title_cache, glove_tools=glove_tools, all_docs=all_docs)
    total_num_matched += num_matched
    all_data.extend(data[i*bsz : min((i+1)*bsz, len(data))])
  # Print stats
  print('=== Summary ===')
  print('Matched %d/%d = %.2f%% questions' % (
      total_num_matched, len(data), 100.0 * total_num_matched / len(data)))
  prefix = '%s-%s' % (OPTS.dataset, adv_strategy)
  if OPTS.prepend:
    prefix = '%s-pre' % prefix
  with open(os.path.join('out', 'hotpot_' + prefix + '.json'), 'w') as f:
    json.dump(all_data, f)


def dump_data_batch(data, adv_strategy, i, bsz=5000):
  if OPTS.add_doc_incl_adv_title:
    filepath = 'data/all_' + OPTS.dataset + '_docs.json'
    all_docs = json.load(open(filepath))
  else:
    all_docs = None

  total_num_matched = 0
  ans_cache, title_cache = None, None
  if adv_strategy == 'addDoc' and 'dyn_gen' in OPTS.rule:
    ans_cache = load_ans_title_cache('answer')
    title_cache = load_ans_title_cache('title')

  glove_tools = None
  if OPTS.find_nearest_glove:
    assert OPTS.dont_replace_full_answer
    glove_tools = build_glove_matrix(num_words=OPTS.num_glove_words_to_use)

  data, num_matched = create_adversarial_exammple(data, adv_strategy, start=i*bsz, end=min((i+1)*bsz, len(data)), 
      ans_cache=ans_cache, title_cache=title_cache, glove_tools=glove_tools, all_docs=all_docs)
  data_batch = data[i*bsz : min((i+1)*bsz, len(data))]
  total_num_matched += num_matched
  # Print stats
  print('=== Summary ===')
  print('Matched %d/%d = %.2f%% questions' % (
      total_num_matched, len(data), 100.0 * total_num_matched / len(data_batch)))
  prefix = '%s-%s' % (OPTS.dataset, adv_strategy)
  if OPTS.prepend:
    prefix = '%s-pre' % prefix
  with open(os.path.join('out', prefix + '-' + str(OPTS.num_new_doc) + '_' + str(i) +'.json'), 'w') as f:
    json.dump(data_batch, f)


def generate_candidate_set(data, out_path, source='answer', bsz=5000):
  ans_set = {}
  for i in range(int(len(data)/bsz) + 1):
    start = i*bsz
    end = min((i+1)*bsz, len(data))
    corenlp_cache = load_cache(start, end)
    for ie, e in tqdm(enumerate(data[start : end])):
      answer, question, context = e['answer'], e['question'], e['context']
      if answer == 'yes' or answer == 'no':
        continue
      _, answer_parse, _, title_parse = corenlp_cache[e['_id']]
      if source == 'answer':
        determiner = get_determiner_for_answer(answer)
        entities = [answer]
        entity_toks = [get_tokens_from_coreobj(answer, answer_parse)]
      else:
        entities = []
        for si, doc in enumerate(context):
          title = doc[0]
          title_split = re.split("([{}])".format("()"), title)
          if title_split[0] != '':
            entities.append(title_split[0])
          elif title_split[-1] != '':
            entities.append(title_split[-1])
          else:
            real_title = title_split[title_split.index('(')+1]
            assert real_title != ')'
            entities.append(real_title)

        entity_toks = [get_tokens_from_coreobj(ent, title_parse[itt]) for itt, ent in enumerate(entities)]

      rules_to_check = FIXED_VOCAB_ANSWER_RULES if source == 'answer' else FIXED_VOCAB_BRIDGE_RULES

      for entity, entity_tok in zip(entities, entity_toks):
        for rule_name, func in rules_to_check:
          if source == 'answer':
            new_entity = func(entity, entity_tok, question, determiner=determiner)
          else:
            new_entity = func(entity, entity_tok, question)

          tok_len = len(entity_tok)
          if new_entity:
            if rule_name not in ans_set:
              ans_set[rule_name] = {}
              ans_set[rule_name][tok_len] = [(entity, entity_tok)]
            else:
              if tok_len in ans_set[rule_name]:
                if (entity, entity_toks) not in ans_set[rule_name][tok_len]:
                  ans_set[rule_name][tok_len].append((entity, entity_tok))
              else:
                ans_set[rule_name][tok_len] = [(entity, entity_tok)]

  for rule in ans_set:
    for tok_len in list(ans_set[rule].keys()):
      ans_set[rule][tok_len] = list(ans_set[rule][tok_len])

  with open(out_path, 'w') as out:
    json.dump(ans_set, out, indent=2)


def generate_all_docs(data, out_path):
  all_docs, all_titles = [], []
  for i, e in enumerate(data):
    for c in e['context']:
      if c[0] not in all_titles:
        all_titles.append(c[0])
        all_docs.append(c)
  with open(out_path, 'w') as f:
    json.dump(all_docs, f)


def merge_adv_examples(out_path):
  if OPTS.dataset == 'train':
    num_files = 19
  else:
    num_files = 2

  adv_data = []
  for i in range(num_files):
    filepath = 'out/' + OPTS.dataset + '-addDoc-4_' + str(i) + '.json'
    with open(filepath, 'r') as f:
      data = json.load(f)
      adv_data += data

  with open(out_path, 'w') as f:
    json.dump(adv_data, f)


def main():
  dataset = read_data()
  if OPTS.command == 'corenlp':
    import re
    run_corenlp(dataset)
  elif OPTS.command == 'dump-addDoc':
    dump_data(dataset, 'addDoc')
  elif OPTS.command == 'dumpBatch-addDoc':
    dump_data_batch(dataset, 'addDoc', OPTS.batch_idx)
  elif OPTS.command == 'gen-answer-set':
    generate_candidate_set(dataset, 'data/' + OPTS.dataset + '_answer_set.json', source='answer')
  elif OPTS.command == 'gen-title-set':
    generate_candidate_set(dataset, 'data/' + OPTS.dataset + '_title_set.json', source='title')
  elif OPTS.command == 'gen-all-docs':
    generate_all_docs(dataset, 'data/' + OPTS.dataset + '_title_set.json')
  elif OPTS.command == 'merge_files':
    merge_adv_examples('out/hotpot_' + OPTS.dataset + '_addDoc.json')
  else:
    raise ValueError('Unknown command "%s"' % OPTS.command)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

