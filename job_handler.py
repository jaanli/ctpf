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
import pmf, hpmf, uaspmf, uaspmf_original, ctpf
import logging
import util
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

parser.add_argument('--trained_user_preferences_file',
  type=str,
  help="trained theta matrix")

parser.add_argument('--out_dir',
  action='store',
  help='directory for output')

# training options
parser.add_argument('--observed_item_attributes_true',
  dest='observed_item_attributes',
  action="store_true",
  help="fix the topic matrix")

parser.add_argument('--observed_item_attributes_false',
  dest='observed_item_attributes',
  action="store_false",
  help="do not fix the topic matrix")

parser.add_argument('--observed_user_preferences_true',
  dest='observed_user_preferences',
  action="store_true",
  help="fix the user prefs matrix")

parser.add_argument('--observed_user_preferences_false',
  dest='observed_user_preferences',
  action="store_false",
  help="do not fix the user prefs matrix")

parser.add_argument('--binarize_true',
  dest='binarize',
  action="store_true",
  help="binarize data" )

parser.add_argument('--binarize_false',
  dest='binarize',
  action='store_false',
  help='do not binarize')

parser.add_argument('--stdout',
  dest='stdout',
  action='store_true',
  help='print to stdout')

parser.add_argument('--resume',
  dest='resume',
  action='store_true',
  help='resume and load')

parser.add_argument('--model',
  type=str,
  help='what model / algo to use')

parser.add_argument('--categorywise_true',
  dest='categorywise',
  action='store_true',
  help='categorywise fit')

parser.add_argument('--categorywise_false',
  dest='categorywise',
  action='store_false',
  help='categorywise fit')

parser.add_argument('--item_fit_type',
  type=str,
  default='all_categories',
  help='how to fit the model, fit options. converge_in_category_first, converge_out_category_first, alternating_updates')

parser.add_argument('--user_fit_type',
  type=str,
  default='default',
  help='''other than default implies initialization by PF fit.

  options: default,
  - means don't update the user prefs, keep them fixed.

  converge_separately,
  - after fitting the item attr, re-fit user prefs after initializing them,
    w/item attr fixed. then re-fit the item attr's and iterate til convergence.

  alternating
  - after fitting item attr, init user prefs, alternate between item_fit_type
  update and updating users til convergence''')

parser.add_argument('--zero_untrained_components_true',
  dest='zero_untrained_components',
  action='store_true',
  default=False,
  help='zero out untrained components')

parser.add_argument('--zero_untrained_components_false',
  dest='zero_untrained_components',
  action='store_false',
  default=False,
  help='categorywise fit')

parser.add_argument('--tolerance',
  type=float,
  default=0.0001,
  help='tolerance for fit')

parser.add_argument('--seed',
  type=int,
  default=98765,
  help='seed')

parser.add_argument('--min_iterations',
  type=int,
  default=1,
  help='minimum number of iterations')

args = parser.parse_args()

# validate arguments
if args.categorywise and args.item_fit_type == 'all_categories':
  raise Exception('need to specify item_fit_type for categorywise!')

if args.item_fit_type == 'alternating_updates' and args.zero_untrained_components:
  raise Exception('cannot zero_untrained_components with alternating_updates!')

if args.item_fit_type == 'alternating_updates' and args.model == 'ctpf':
  raise Exception('unsupported alternating updates with ctpf currently')

if args.observed_user_preferences and not args.trained_user_preferences_file:
  raise Exception('need trained user preferences to fix user prefs')

if not os.path.exists(args.out_dir):
  os.makedirs(args.out_dir)

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
if args.stdout:
  logger.addHandler(ch)

for arg, value in sorted(vars(args).items()):
  logger.info("{}: {}".format(arg, value))

logger.info('=>loading metadata')
id2arxiv_info = pd.read_csv(args.item_info_file, header=None, delimiter='\t', names=['arxiv_id', 'categories', 'title', 'date'])
unique_did = list(id2arxiv_info.index)
n_docs = np.unique(unique_did).shape[0]

if args.observed_item_attributes or args.categorywise:
  document_category_dummies = id2arxiv_info['categories'].str.join(sep='').str.get_dummies(sep=' ')
  category_list = list(document_category_dummies.columns)
  n_categories = len(category_list)
  #logging.info('observed topics => num categories (k) = {}'.format(n_categories))
  observed_categories = document_category_dummies.as_matrix().astype(np.float32)
  # check if we have zeros in all rows for some docs
  assert len(np.where(~observed_categories.any(axis=1))[0]) == 0
else:
  n_categories = 166
  observed_categories = False

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

if args.trained_user_preferences_file:
  h5f_t = h5py.File(args.trained_user_preferences_file)
  Et_t = h5f_t['Et_t'][:]
  Et_loaded = Et_t.T
  print Et_loaded[0:100, 0:100]
  logger.info('loaded trained user preferences from fit file')
else:
  Et_loaded = False

logger.info('=>running fit')

h5f = h5py.File('{}fit.h5'.format(args.out_dir), 'w')

if args.model == 'pmf':
  coder = pmf.PoissonMF(n_components=n_categories, random_state=args.seed,
    verbose=True, a=0.1, b=0.1, c=0.1, d=0.1, logger=logger, tol=args.tolerance,
    min_iter=args.min_iterations)
  if args.resume:
    Eb_t = h5f['Eb_t'][:]
    Et_t = h5f['Et_t'][:]
    logging.info('loaded fit!')
  else:
    if args.observed_item_attributes:
        coder.fit(train_data, rows, cols, validation, beta=observed_categories,
          theta=Et_loaded, user_fit_type=args.user_fit_type,
          categorywise=args.categorywise, item_fit_type=args.item_fit_type,
          zero_untrained_components=args.zero_untrained_components)
    else:
      coder.fit(train_data, rows, cols, validation)

    Et_t = np.ascontiguousarray(coder.Et.T)
    Eb_t = np.ascontiguousarray(coder.Eb.T)
    h5f.create_dataset('Eb_t', data=Eb_t)
    h5f.create_dataset('Et_t', data=Et_t)

elif args.model == 'ctpf':
  song2artist = np.array([n for n in range(n_docs)])
  # first fit vanilla poisson factorization for user preferences
  hyper = 0.3
  coder = ctpf.PoissonMF(n_components=n_categories, smoothness=100,
      max_iter=8, random_state=98765, verbose=True,
      a=hyper, b=hyper, c=hyper, d=hyper, f=hyper, g=hyper, s2a=song2artist,
      min_iter=args.min_iterations,
      beta=observed_categories,
      theta=Et_loaded,
      categorywise=args.categorywise,
      item_fit_type=args.item_fit_type,
      user_fit_type=args.user_fit_type,
      observed_item_attributes=args.observed_item_attributes,
      observed_user_preferences=args.observed_user_preferences,
      zero_untrained_components=args.zero_untrained_components)
  if args.resume:
    Eba_t = h5f['Eba_t'][:]
    Ebs_t = h5f['Ebs_t'][:]
    Et_t = h5f['Et_t'][:]
    Eb_t = Ebs_t + Eba_t
    logging.info('loaded fit!')
  else:
    if args.observed_item_attributes:
      # run vanilla PF to get user prefs first
      # coder_pmf = pmf.PoissonMF(n_components=n_categories, random_state=98765,
      #   verbose=True, a=0.1, b=0.1, c=0.1, d=0.1, logger=logger)
      # coder_pmf.fit(train_data, rows, cols, validation, beta=observed_categories)

      # # calc log-likelihood of this
      # util.calculate_loglikelihood(coder_pmf, train, validation, test)

      # fit ctpf with fixed user prefs and observed topics
      # item_fit_type = 'default': just fit epsilons normally.
      # item_fit_type = alternating: update in_category components, then out_category components.
      # item_fit_type = converge_in_category_components first:
      coder.fit(train_data, rows, cols, validation)
    else:
      # just run vanilla ctpf
      coder.fit(train_data, rows, cols, validation)

    Et_t = np.ascontiguousarray(coder.Et.T)
    Eb_t = np.ascontiguousarray(coder.Eb.T)
    Eeps_t = np.ascontiguousarray(coder.Eeps.T)
    Eb_t = Eb_t + Eeps_t
    h5f.create_dataset('Et_t', data=Et_t)
    h5f.create_dataset('Eb_t', data=Eb_t)
    h5f.create_dataset('Eeps_t', data=Eeps_t)

elif args.model == 'ctpf_original':
  song2artist = np.array([n for n in range(n_docs)])
  # first fit vanilla poisson factorization for user preferences
  hyper = 0.3
  coder = uaspmf_original.PoissonMF(n_components=n_categories, smoothness=100,
      max_iter=8, random_state=98765, verbose=True,
      a=hyper, b=hyper, c=hyper, d=hyper, f=hyper, g=hyper, s2a=song2artist)
  if args.resume:
    Eba_t = h5f['Eba_t'][:]
    Ebs_t = h5f['Ebs_t'][:]
    Et_t = h5f['Et_t'][:]
    Eb_t = Ebs_t + Eba_t
    logging.info('loaded fit!')
  else:
      coder.fit(train_data, rows, cols, validation)

  Et_t = np.ascontiguousarray(coder.Et.T)
  Eba_t = np.ascontiguousarray(coder.Eba.T)
  Ebs_t = np.ascontiguousarray(coder.Ebs.T)
  Eb_t = Ebs_t + np.ascontiguousarray(coder.Eba[song2artist].T)
  h5f.create_dataset('Et_t', data=Et_t)
  h5f.create_dataset('Eba_t', data=Eba_t)
  h5f.create_dataset('Ebs_t', data=Ebs_t)

elif args.model == 'hpmf':
  coder = hpmf.HPoissonMF(n_components=n_categories, max_iter=500,
    random_state=98765, verbose=True, min_iter=args.min_iterations,
    a=0.3, c=0.3, a_ksi=0.3, b_ksi=0.3, c_eta=0.3, d_eta=0.3)
  if args.resume:
    Eb_t = h5f['Eb_t'][:]
    Et_t = h5f['Et_t'][:]
    logging.info('loaded fit!')
  else:
    if args.observed_item_attributes:
      coder.fit(train_data, rows, cols, validation, beta=observed_categories,
        categorywise=args.categorywise, item_fit_type=args.item_fit_type,
        zero_untrained_components=args.zero_untrained_components)
    else:
      coder.fit(train_data, rows, cols, validation)
    Et_t = np.ascontiguousarray(coder.Et.T)
    Eb_t = np.ascontiguousarray(coder.Eb.T)
    h5f.create_dataset('Eb_t', data=Eb_t)
    h5f.create_dataset('Et_t', data=Et_t)
h5f.close()

if not args.resume:
  # print coder.Eb
  # print '^eb'
  # print coder.Et
  # print '^et'
  util.calculate_loglikelihood(coder, train, validation, test)

rec_eval.calc_all(train_data, validation_smat, test_smat, Et_t, Eb_t)


