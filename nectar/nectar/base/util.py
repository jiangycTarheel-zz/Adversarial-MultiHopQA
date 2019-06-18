"""General, miscellaneous utilities."""
from contextlib import contextmanager
import sys
import time

def flatten(x):
  """Flatten a list of lists."""
  return [a for b in x for a in b]

def log(msg, disappearing=False):
  if not sys.stdout.isatty():
    # Only print to stdout if it's being redirected or piped
    print(msg)
  # if disappearing:
  #   # Trailing comma suppresses newline
  #   print >> sys.stderr, msg + '\r',
  # else:
  #   print >> sys.stderr, msg

def log_dict(d, name):
  log('%s {' % name)
  for k in d:
    log('    %s: %s' % (k, str(d[k])))
  log('}')

SECS_PER_MIN = 60
SECS_PER_HOUR = SECS_PER_MIN * 60
SECS_PER_DAY = SECS_PER_HOUR * 24

def secs_to_str(secs):
  """Convert a number of seconds to human-readable string."""
  days = int(secs) / SECS_PER_DAY
  secs -= days * SECS_PER_DAY
  hours = int(secs) / SECS_PER_HOUR
  secs -= hours * SECS_PER_HOUR
  mins = int(secs) / SECS_PER_MIN
  secs -= mins * SECS_PER_MIN
  if days > 0:
    return '%dd%02dh%02dm' % (days, hours, mins)
  elif hours > 0:
    return '%dh%02dm%02ds' % (hours, mins, int(secs))
  elif mins > 0:
    return '%dm%02ds' % (mins, int(secs))
  elif secs >= 1:
    return '%.1fs' % secs
  return '%.2fs' % secs

def timed(func, msg, allow_overwrite=True):
  msg1 = '%s...' % msg
  log(msg1, disappearing=allow_overwrite)
  t0 = time.time()
  retval = func()
  t1 = time.time()
  msg2 = '%s [took %s].' % (msg, secs_to_str(t1 - t0))
  log(msg2)
  return retval

@contextmanager
def timer(msg, allow_overwrite=True):
  msg1 = '%s...' % msg
  log(msg1, disappearing=allow_overwrite)
  t0 = time.time()
  yield
  t1 = time.time()
  msg2 = '%s [took %s].' % (msg, secs_to_str(t1 - t0))
  log(msg2)
