# this script handles loading, preprocessing of data and launching experiments for:
# vanilla PF, hierarchical PF, user-artist-song PF / observed CTPF
# input: command line arguments specifying train/validation/test files and the algorithm to use
# output: held-out evaluation metrics, training log-likelihood file, saved user preferences / epsilons

import argparse
import sys
import rec_eval
import pandas as pd
import numpy as np
import scipy
import pmf
import logging
import util
import uaspmf
import h5py
import os

parser = argparse.ArgumentParser(description='this script handles loading, preprocessing of data and launching experiments')

# io options
parser.add_argument('--train_file',
    type=argparse.FileType('r'),
    help="train tsv")

parser.add_argument('--validation_file',
    type=argparse.FileType('r'),
    help="validation tsv")

parser.add_argument('--test_file',
    type=argparse.FileType('r'),
    help="test tsv")

parser.add_argument('--item_info_file',
    type=argparse.FileType('r'),
    help="info on items, IDs")

parser.add_argument('--user_info_file',
  type=argparse.FileType('r'),
  help="info on users, IDs")

parser.add_argument('--out_dir',
  action='store',
  help='directory for output')

# training options
parser.add_argument('--observed_topics_true',
  dest='observed_topics',
  action="store_true",
  help="fix the topic matrix")

parser.add_argument('--observed_topics_false',
  dest='observed_topics',
  action="store_false",
  help="do not fix the topic matrix")

parser.add_argument('--binarize_true',
  dest='binarize',
  action="store_true",
  help="binarize data" )

parser.add_argument('--binarize_false',
  dest='binarize',
  action='store_false',
  help='do not binarize')


parser.add_argument('--model',
  type=str,
  help='what model / algo to use')

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(args.out_dir + 'job.log', 'w')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s\t %(levelname)s L%(lineno)s\t %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

for arg, value in sorted(vars(args).items()):
  logger.info("{}: {}".format(arg, value))

logger.info('=>loading metadata')
id2arxiv_info = pd.read_csv(args.item_info_file, header=None, delimiter='\t', names=['arxiv_id', 'categories', 'title', 'date'])
unique_did = list(id2arxiv_info.index)
n_docs = np.unique(unique_did).shape[0]

if args.observed_topics:
  document_category_dummies = id2arxiv_info['categories'].str.join(sep='').str.get_dummies(sep=' ')
  category_list = list(document_category_dummies.columns)
  n_categories = len(category_list)
  #logging.info('observed topics => num categories (k) = {}'.format(n_categories))
  observed_categories = document_category_dummies.as_matrix().astype(np.float32)
  # check if we have zeros in all rows for some docs
  assert len(np.where(~observed_categories.any(axis=1))[0]) == 0
else:
  n_categories = 166

logger.info('number of categories is k={}'.format(n_categories))

# load num_users
id2arxiv_uid = pd.read_csv(args.user_info_file, header=None, delimiter='\t', names=['uid'])
n_users = np.unique(id2arxiv_uid.uid).shape[0]

logger.info('num docs is {}, num users is {}'.format(n_docs, n_users))


logger.info('=>loading data')
train_data, rows, cols = rec_eval.load_data(
  args.train_file, (n_docs, n_users), args.binarize)
train = dict(X_new=train_data.data, rows_new=rows, cols_new=cols)

validation_smat, rows_validation, cols_validation = rec_eval.load_data(
  args.validation_file, (n_docs, n_users), args.binarize)
validation = dict(X_new=validation_smat.data,
           rows_new=rows_validation,
           cols_new=cols_validation)

test_smat, rows_test, cols_test = rec_eval.load_data(args.test_file,
  (n_docs, n_users), args.binarize)
test = dict(X_new=test_smat.data,
  rows_new=rows_test,
  cols_new=cols_test)

logger.info('=>running fit')

h5f = h5py.File('{}fit.h5'.format(args.out_dir), 'w')

if args.model == 'pmf':
  coder = pmf.PoissonMF(n_components=n_categories, random_state=98765,
    verbose=True, a=0.1, b=0.1, c=0.1, d=0.1, logger=logger)
  if args.observed_topics:
    coder.fit(train_data, rows, cols, validation, beta=observed_categories)
  else:
    coder.fit(train_data, rows, cols, validation)

  Et_t = np.ascontiguousarray(coder.Et.T)
  Eb_t = np.ascontiguousarray(coder.Eb.T)
  h5f.create_dataset('Eb_t', data=Eb_t)
  h5f.create_dataset('Et_t', data=Et_t)

elif args.model == 'ctpf':
  song2artist = np.array([n for n in range(n_docs)])
  coder = uaspmf.PoissonMF(n_components=n_categories, smoothness=100,
      max_iter=100, random_state=98765, verbose=True,
      a=0.3, b=0.3, c=0.3, d=0.3, f=0.3, g=0.3, s2a=song2artist)

  if args.observed_topics:
    # run vanilla PF to get user prefs first
    coder_pmf = pmf.PoissonMF(n_components=n_categories, random_state=98765,
      verbose=True, a=0.1, b=0.1, c=0.1, d=0.1, logger=logger)
    coder_pmf.fit(train_data, rows, cols, validation, beta=observed_categories)

    # calc log-likelihood of this
    util.calculate_loglikelihood(coder_pmf, train, validation, test)

    # fit ctpf with fixed user prefs and observed topics
    coder.fit(train_data, rows, cols, validation, beta=observed_categories,
      theta=coder_pmf.Et)
  else:
    coder.fit(train_data, rows, cols, validation)

  Et_t = np.ascontiguousarray(coder.Et.T)
  Eba_t = np.ascontiguousarray(coder.Eba.T)
  Ebs_t = np.ascontiguousarray(coder.Ebs.T)
  Eb_t = Ebs_t + np.ascontiguousarray(coder.Eba[song2artist].T)
  h5f.create_dataset('Et_t', data=Et_t)
  h5f.create_dataset('Eba_t', data=Eba_t)
  h5f.create_dataset('Ebs_t', data=Ebs_t)

elif args.model == 'categorywise':
  #todo
  pass

elif args.model == 'hier_ctpf':
  #todo
  pass

h5f.close()

util.calculate_loglikelihood(coder, train, validation, test)

#rec_eval.calc_all(train_data, validation_smat, test_smat, Et_t, Eb_t)


