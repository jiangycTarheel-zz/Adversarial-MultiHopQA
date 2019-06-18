import math
import random
import string 
import json
import numpy as np
from pattern.en import conjugate
from resources.nectar import corenlp
from nltk.corpus import wordnet
from nltk.corpus import stopwords

COMMON_WORDS = list(stopwords.words('english'))
PUNCTUATIONS = list(string.punctuation)


def ans_number(a, tokens, q, **kwargs):
  out_toks = []
  seen_num = False
  for t in tokens:
    ner = t['ner']
    pos = t['pos']
    w = t['word']
    out_tok = {'before': t['before']}

    # Split on dashes
    leftover = ''
    dash_toks = w.split('-')
    if len(dash_toks) > 1:
      w = dash_toks[0]
      leftover = '-'.join(dash_toks[1:])

    # Try to get a number out
    value = None
    if w != '%': 
      # Percent sign should just pass through
      try:
        value = float(w.replace(',', ''))
      except:
        try:
          norm_ner = t['normalizedNER']
          if norm_ner[0] in ('%', '>', '<'):
            norm_ner = norm_ner[1:]
          value = float(norm_ner)
        except:
          pass
    if not value and (
        ner == 'NUMBER' or 
        (ner == 'PERCENT' and pos == 'CD')):
      # Force this to be a number anyways
      value = 10
    if value:
      if math.isinf(value) or math.isnan(value): value = 9001
      seen_num = True
      if w in ('thousand', 'million', 'billion', 'trillion'):
        if w == 'thousand':
          new_val = 'million'
        else:
          new_val = 'thousand'
      else:
        if value < 2500 and value > 1000:
          rand = np.random.randint(0, 10)
          if rand%2 == 0:
            new_val = str(value - np.random.randint(1, 11))
          else:
            new_val = str(value + np.random.randint(1, 11))
        else:
          # Change leading digit
          if value == int(value):
            val_chars = list('%d' % value)
          else:
            val_chars = list('%g' % value)
          c = val_chars[0]
          for i in range(len(val_chars)):
            c = val_chars[i]
            if c >= '0' and c <= '9':
              val_chars[i] = str(max((int(c) + np.random.randint(1, 10)) % 10, 1))
              break
          new_val = ''.join(val_chars)
      if leftover:
        new_val = '%s-%s' % (new_val, leftover)
      out_tok['originalText'] = new_val
    else:
      out_tok['originalText'] = t['originalText']
    
    if t['originalText'].endswith('.0') is False and out_tok['originalText'].endswith('.0'):
      out_tok['originalText'] = out_tok['originalText'][:-2]

    out_toks.append(out_tok)
  if seen_num:
    return corenlp.rejoin(out_toks).strip()
  else:
    return None


MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']


def ans_date(a, tokens, q, **kwargs):
  out_toks = []
  if not all(t['ner'] == 'DATE' for t in tokens): return None
  for t in tokens:
    if t['pos'] == 'CD' or t['word'].isdigit():
      try:
        value = int(t['word'])
      except:
        value = 10  # fallback
      if value > 50:  
        rand = np.random.randint(0, 10)
        if rand%2 == 0:
          new_val = str(value - np.random.randint(10, 25))  # Year
        else:
          new_val = str(value + np.random.randint(10, 25))  # Year
      else:  # Day of month
        if value > 15: new_val = str(value - np.random.randint(1, 12))
        else: new_val = str(value + np.random.randint(1, 12))
    else:
      if t['word'].lower() in MONTHS:
        m_ind = MONTHS.index(t['word'].lower())
        new_val = MONTHS[(m_ind + np.random.randint(1, 11)) % 12].title()
      else:
        # Give up
        new_val = t['originalText']
    out_toks.append({'before': t['before'], 'originalText': new_val})
  new_ans = corenlp.rejoin(out_toks).strip()
  if new_ans == a: return None
  return new_ans


def lookup_answer_generate(checker, rule):
  '''
  uses cached list of old answers to generate a real answer, return None if it
  does not pass the checker function
  '''
  def func(a, tokens, question, ans_cache, **kwargs):
    fake = checker(a, tokens, question, **kwargs)
    if fake is None:
      return None

    tok_len = len(tokens)    
    counter = 0
    new_ans = a
    while new_ans == a:
      if tok_len <= 0:
        return None
      counter2 = 0
      while True:
        if str(tok_len) in ans_cache[rule]:
          new_ans, new_ans_tok = random.choice(ans_cache[rule][str(tok_len)])
          if a.lower().startswith('the ') and (new_ans.lower().startswith('the ') is False) \
          or a.lower().startswith('the ') is False and new_ans.lower().startswith('the '):
            counter2 += 1
            if counter2 == 40:
              break
          else:
            break
        else:
          tok_len -= 1
          if tok_len <= 0:
            return None

      counter += 1
      if counter == 40:
        tok_len -= 1
        counter = 0

    if a.lower().startswith('the ') and (new_ans.lower().startswith('the ') is False):
      new_ans = 'The ' + new_ans
      new_ans_tok = [{'originalText': 'The', 'pos': 'DT', 'word': 'the'}] + new_ans_tok
      assert counter2 == 40

    return new_ans, new_ans_tok
  return func


FIRST_NAMES = ['Jason', 'Mary', 'James', 'Jeff', 'Abi', 'Bran', 'Sansa', 'Jon', 'Ned', 'Peter', 'Jaime', \
'Marcus', 'Chris', 'Diana', 'Phoebe', 'Leo', 'Phil', 'Nick', 'Steve']
LAST_NAMES = ['Kid', 'Jordan', 'Harden', 'Dean', 'Stark', 'Parker', 'Morris', 'Wallace', 'Manning', 'Rogers', 'Folt', 'White']
LOCATIONS = ['Chicago', 'New York', 'Beijing', 'Tokyo', 'Pittsburg', 'Los Angeles', 'Paris', 'Barcelona', 'Madrid', 'Berlin']
ORGANIZATIONS = ['Stark Industries', 'Google Inc', 'Baidu Inc', 'Nike Corp', 'House of Stark', 'University of Southern Texas', \
'National Student Association', 'Facebook', 'Department of Education']
NNP = ['Central Park', 'Student Store', 'White House', 'Pacific Ocean', 'Gourmet Center', 'Golden Palace', 'Stony River', 'Staples Center']
NNPS = ['Cool Kids', 'Kew Gardens', 'Silver Bullets', 'LA Lakers', 'Brooks Brothers']
NN = ['hamster', 'composer', 'man', 'statement']
NNS = ['hamsters', 'composers', 'men', 'statements']


def process_token(word, original_tok):
  new_word = word
  if original_tok['pos'].startswith('V'):
    if original_tok['pos'] == 'VB':
      new_word = conjugate(word, 'VB')
    elif original_tok['pos'] == 'VBD':
      new_word = conjugate(word, 'VBD')
    elif original_tok['pos'] == 'VBN':
      new_word = conjugate(word, 'VBN')
    elif original_tok['pos'] == 'VBG':
      new_word = conjugate(word, 'VBG')
    elif original_tok['pos'] == 'VBZ':
      new_word = conjugate(word, 'VBZ')
    elif original_tok['pos'] == 'VBP':
      new_word = conjugate(word, 'VBP')
  return new_word


def ans_wordnet_catch_amap(a, tokens, q, **kwargs):
  new_ans, new_ans_tok = [], []
  for t in tokens:
    if t['originalText'].lower() in COMMON_WORDS + PUNCTUATIONS:
      new_ans.append(t['originalText'])
      new_ans_tok.append(t)
      continue 
    synonyms, antonyms = [], [] 
    for syn in wordnet.synsets(t['originalText']): 
      for l in syn.lemmas(): 
        synonyms.append(l.name()) 
        if l.antonyms(): 
            antonyms.append(l.antonyms()[0].name())
    
    new_word = None
    if t['pos'].startswith('VB') or t['pos'].startswith('JJ') or t['pos'].startswith('R'):
      for w in antonyms:
        if w.lower() != t['originalText'].lower()  and t['originalText'] not in w.lower() and '_' not in w:
          new_word = process_token(w, t)
          break
    if new_word is None:
      for w in synonyms:
        if w.lower() not in t['originalText'].lower() and t['originalText'] not in w.lower() and '_' not in w:
          new_word = process_token(w, t)
          break
    if new_word and new_word not in t['originalText'] and t['originalText'] not in new_word:
      new_ans.append(new_word)
      new_ans_tok.append({'originalText': new_word, 'pos': t['pos']})
    else:
      return None
  new_ans = ' '.join(new_ans)
  if new_ans.lower() == a.lower():
    return None
  return new_ans, new_ans_tok


def ans_entity_full(ner_tag, new_ans):
  """Returns a function that yields new_ans iff every token has |ner_tag|."""
  def func(a, tokens, q, **kwargs):
    for t in tokens:
      if t['ner'] != ner_tag: return None
    if ner_tag == 'PERSON':
      fname = FIRST_NAMES[random.randint(0, len(FIRST_NAMES)-1)]
      lname = LAST_NAMES[random.randint(0, len(LAST_NAMES)-1)]
      return fname + ' ' + lname 
    elif ner_tag == 'LOCATIONS':
      return LOCATIONS[random.randint(0, len(LOCATIONS)-1)]
    elif ner_tag == 'ORGANIZATION':
      return ORGANIZATIONS[random.randint(0, len(ORGANIZATIONS)-1)]
    return new_ans
  return func


def ans_abbrev(new_ans):
  def func(a, tokens, q, **kwargs):
    s = a
    if s == s.upper() and s != s.lower():
      return new_ans
    return None
  return func


def ans_match_wh(wh_word, new_ans):
  """Returns a function that yields new_ans if the question starts with |wh_word|."""
  def func(a, tokens, q, **kwargs):
    if q.lower().startswith(wh_word + ' '):
      if wh_word == 'who':
        fname = FIRST_NAMES[random.randint(0, len(FIRST_NAMES)-1)]
        lname = LAST_NAMES[random.randint(0, len(LAST_NAMES)-1)]
        return fname + ' ' + lname 
      elif wh_word == 'where':
        return LOCATIONS[random.randint(0, len(LOCATIONS)-1)]
      return new_ans
    return None
  return func


def ans_pos(pos, new_ans, end=False, add_dt=False):
  """Returns a function that yields new_ans if the first/last token has |pos|."""
  def func(a, tokens, q, determiner, **kwargs):
    if end:
      for it in range(len(tokens)):
        t = tokens[-1-it]
        if t['originalText'] not in PUNCTUATIONS:
          break
    else:
      t = tokens[0]
    if t['pos'] != pos: return None
    if add_dt and determiner:
      if pos == 'NN':
        return '%s %s' % (determiner, NN[random.randint(0, len(NN)-1)])
      if pos == 'NNS':
        return '%s %s' % (determiner, NNS[random.randint(0, len(NNS)-1)])
      if pos == 'NNP':
        return '%s %s' % (determiner, NNP[random.randint(0, len(NNP)-1)])
      if pos == 'NNPS':
        return '%s %s' % (determiner, NNPS[random.randint(0, len(NNPS)-1)])
      return '%s %s' % (determiner, new_ans)
    if pos == 'NN':
      return NN[random.randint(0, len(NN)-1)]
    if pos == 'NNS':
      return NNS[random.randint(0, len(NNS)-1)]
    if pos == 'NNP':
      return NNP[random.randint(0, len(NNP)-1)]
    if pos == 'NNPS':
      return NNPS[random.randint(0, len(NNPS)-1)]
    return new_ans
  return func

  
def ans_catch_all(new_ans):
  def func(a, tokens, q, **kwargs):
    if tokens[0]['originalText'][0].isupper():
      return new_ans[0].upper()+new_ans[1:]
    return new_ans
  return func