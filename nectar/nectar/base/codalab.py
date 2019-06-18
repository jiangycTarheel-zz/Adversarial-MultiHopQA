"""Utilities for interacting with codalab."""
import collections
import subprocess
import sys

from .. import log

DOCKER_IMAGE = 'robinjia/robinjia-codalab:2.1.1'

def run(cmd, deps, name, description, queue='john', host=None, cpus=1,
        docker_image=DOCKER_IMAGE, is_theano=False, omp_num_threads=1,
        dry_run=False):
  params = collections.OrderedDict()
  if host:
    params['--request-queue'] = 'host=%s' % host
  else:
    params['--request-queue'] = queue
  params['--request-cpus'] = str(cpus)
  params['--request-docker-image'] = docker_image
  params['-n'] = name
  params['-d'] = description
  param_list = [x for k_v in params.iteritems() for x in k_v]
  if is_theano:
    prefix = 'OMP_NUM_THREADS=%d THEANO_FLAGS=blas.ldflags=-lopenblas' % omp_num_threads
    cmd = prefix + ' ' + cmd
  call_args = ['cl', 'run'] + deps + [cmd] + param_list
  if dry_run:
    log('Dry run: %s' % str(call_args))
  else:
    subprocess.call(call_args)

def upload(filename, name=None, description=None, dry_run=False):
  call_args = ['cl', 'up', filename]
  if name:
    call_args.extend(['-n', name])
  if description:
    call_args.extend(['-d', description])
  if dry_run:
    log('Dry run: %s' % str(call_args))
  else:
    subprocess.call(call_args)
