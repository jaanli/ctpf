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
trained_user_preferences_file = '/home/waldorf/altosaar/projects/arxiv/fit/2015-11-3-pmf-binarize_true-observed_topics_true/fit.h5'


parameters = dict(model = ['pmf', 'ctpf'],
  binarize = ['binarize_true'],
  observed_topics = ['observed_topics_true'],
  categorywise = ['categorywise_true'],
  item_fit_type = ['converge_in_category_first', 'converge_out_category_first', 'alternating_updates'],
  user_fit_type = ['converge_separately'],
  zero_untrained_components = ['zero_untrained_components_false'],
  train_file = [train_file],
  validation_file = [validation_file],
  test_file = [test_file],
  item_info_file = [item_info_file],
  user_info_file = [user_info_file],
  trained_user_preferences_file = [trained_user_preferences_file],
  min_iterations = ['3'],
  stdout = ['stdout'],
  #resume = ['resume']
  )

# create timestamped directory in fit
now = time.localtime()[0:3]
base_dir_name = out_dir + '{}-{}-{}_best_ll'.format(now[0], now[1], now[2])

for setting in dict_product(parameters):

  out_dir_path = '{}-{}-{}-{}/'.format(base_dir_name,
    setting['model'],
    setting['categorywise'],
    setting['item_fit_type'])

  setting['out_dir'] = out_dir_path

  setting_list = []
  for k, v in setting.items():
    if k not in ['binarize', 'observed_topics', 'stdout', 'resume', 'zero_untrained_components', 'categorywise']:
      setting_list += ['--' + k]
      setting_list += [v]
    else:
      setting_list += ['--' + v]

  print setting_list

  subprocess.Popen(['python', 'job_handler.py'] + setting_list)

  print 'launched job & created directory {}'.format(out_dir_path)
