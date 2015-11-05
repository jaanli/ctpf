# use run.sh to launch many jobs for all possible settings on the toy dataset

import subprocess
from itertools import product, izip
import os, time
import logging

#in_dir = '/home/statler/lcharlin/arxiv/dat/dataset_toy/'
in_dir = '/home/statler/lcharlin/arxiv/dat/dataset_2003-2012_clean/'
out_dir = '/home/waldorf/altosaar/projects/arxiv/fit/'

def dict_product(dicts):
    return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))

train_file = in_dir + 'train.tsv'
validation_file = in_dir + 'validation.tsv'
test_file = in_dir + 'test.tsv'
item_info_file = in_dir + 'items_arxiv_info.tsv'
user_info_file = in_dir + 'users.tsv'

parameters = dict(model = ['pmf', 'ctpf'],
  binarize = ['binarize_true', 'binarize_false'],
  observed_topics = ['observed_topics_true', 'observed_topics_false'],
  train_file = [train_file],
  validation_file = [validation_file],
  test_file = [test_file],
  item_info_file = [item_info_file],
  user_info_file = [user_info_file],
  stdout = ['stdout'],
  #resume = ['resume']
  )

# create timestamped directory in fit
now = time.localtime()[0:3]
base_dir_name = out_dir + '{}-{}-{}'.format(now[0], now[1], now[2])

for setting in dict_product(parameters):

  out_dir_path = '{}-{}-{}-{}/'.format(base_dir_name, setting['model'], setting['binarize'], setting['observed_topics'])
  if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)

  setting['out_dir'] = out_dir_path

  setting_list = []
  for k, v in setting.items():
    if k not in ['binarize', 'observed_topics', 'stdout', 'resume']:
      setting_list += ['--' + k]
      setting_list += [v]
    else:
      setting_list += ['--' + v]

  subprocess.Popen(['python', 'job_handler.py'] + setting_list)

  print 'launched job & created directory {}'.format(out_dir_path)
