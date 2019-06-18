"""Add standard theano-related flags to an argparse.ArgumentParser."""
import argparse
import sys
import theano

from .. import log, log_dict

class NLPArgumentParser(argparse.ArgumentParser):
  """An ArgumentParser with some built-in arguments.
  
  Allows you to not have to retype help messages every time.
  """
  def add_flag_helper(self, long_name, short_name, *args, **kwargs):
    long_flag = '--%s' % long_name
    if 'help' in kwargs:
      if 'default' in kwargs:
        # Append default value to help message
        new_help = '%s (default=%s)' % (kwargs['help'], str(kwargs['default']))
        kwargs['help'] = new_help
      # Append period to end, if missing
      if not kwargs['help'].endswith('.'):
        kwargs['help'] = kwargs['help'] + '.'
    if short_name:
      short_flag = '-%s' % short_name
      self.add_argument(long_flag, short_flag, *args, **kwargs)
    else:
      self.add_argument(long_flag, *args, **kwargs)

  # Model hyperparameters
  def add_hidden_size(self, short_name=None):
    self.add_flag_helper('hidden-size', short_name, type=int,
                         help='Dimension of hidden vectors')
  def add_emb_size(self, short_name=None):
    self.add_flag_helper('emb-size', short_name, type=int, 
                         help='Dimension of word vectors')
  def add_weight_scale(self, short_name=None, default=1e-1):
    self.add_flag_helper('weight-scale', short_name, type=float, default=default,
                         help='Weight scale for initialization')
  def add_l2_reg(self, short_name=None, default=0.0):
    self.add_flag_helper('l2-reg', short_name, type=float, default=default,
                         help='L2 Regularization constant.')
  def add_unk_cutoff(self, short_name=None):
    self.add_flag_helper('unk-cutoff', short_name, type=int, default=0,
                         help='Treat input words with <= this many occurrences as UNK')

  # Training hyperparameters
  def add_num_epochs(self, short_name=None):
    self.add_flag_helper(
        'num-epochs', short_name, default=[], type=lambda s: [int(x) for x in s.split(',')], 
        help=('Number of epochs to train. If comma-separated list, will run for some epochs, halve learning rate, etc.'))
  def add_learning_rate(self, short_name=None, default=0.1):
    self.add_flag_helper('learning-rate', short_name, type=float, default=default,
                         help='Initial learning rate.')
  def add_clip_thresh(self, short_name=None):
    self.add_flag_helper('clip-thresh', short_name, type=float, default=1.0,
                         help='Total-norm threshold to clip gradients.')
  def add_batch_size(self, short_name=None):
    self.add_flag_helper('batch-size', short_name, type=int, default=1,
                         help='Maximum batch size')
  # Decoding hyperparameters
  def add_beam_size(self, short_name=None):
    self.add_flag_helper('beam-size', short_name, type=int, default=0,
                         help='Use beam search with given beam size, or greedy if 0')
  # Data
  def add_train_file(self, short_name=None):
    self.add_flag_helper('train-file', short_name, help='Path to training data')
  def add_dev_file(self, short_name=None):
    self.add_flag_helper('dev-file', short_name, help='Path to dev data')
  def add_test_file(self, short_name=None):
    self.add_flag_helper('test-file', short_name, help='Path to test data')
  def add_dev_frac(self, short_name=None):
    self.add_flag_helper('dev-frac', short_name, type=float, default=0.0,
                         help='Take this fraction of train data as dev data')

  # Random seeds
  def add_dev_seed(self, short_name=None):
    self.add_flag_helper('dev-seed', short_name, type=int, default=0,
                         help='RNG seed for the train/dev splits')
  def add_model_seed(self, short_name=None):
    self.add_flag_helper('model-seed', short_name, type=int, default=0,
                         help="RNG seed for the model")

  # Sasving and loading
  def add_save_file(self, short_name=None):
    self.add_flag_helper('save-file', short_name, help='Path for saving model')
  def add_load_file(self, short_name=None):
    self.add_flag_helper('load-file', short_name, help='Path for loading model')

  # Output
  def add_stats_file(self, short_name=None):
    self.add_flag_helper('stats-file', short_name, help='File to write stats JSON')
  def add_html_file(self, short_name=None):
    self.add_flag_helper('html-file', short_name, help='File to write output HTML')

  def add_theano_flags(self):
    self.add_flag_helper('theano-fast-compile', None, action='store_true', help='Run Theano in fast compile mode')
    self.add_flag_helper('theano-profile', None, action='store_true', help='Turn on profiling in Theano')

  def parse_args(self):
    """Configure theano and print help on empty arguments."""
    if len(sys.argv) == 1:
      self.print_help()
      sys.exit(1)
    args = super(NLPArgumentParser, self).parse_args()
    log_dict(vars(args), 'Command-line Arguments')
    configure_theano(args)
    return args


def configure_theano(opts):
  """Configure theano given arguments passed in."""
  if opts.theano_fast_compile:
    theano.config.mode='FAST_COMPILE'
    theano.config.optimizer = 'None'
    theano.config.traceback.limit = 20
  else:
    theano.config.mode='FAST_RUN'
    theano.config.linker='cvm'
  if opts.theano_profile:
    theano.config.profile = True
