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
parser.add_argument('--observed_topics',
  action="store_true",
  help="fix the topic matrix")

parser.add_argument('--binarize',
  action="store_true",
  help="binarize data" )

parser.add_argument('--debug',
  action='store_true',
  help='debug mode')

args = parser.parse_args()


print '=> loading metadata'
id2arxiv_info = pd.read_csv(args.item_info_file, header=None, delimiter='\t', names=['arxiv_id', 'categories', 'title', 'date'])

# drop articles with no category
drop_indices = id2arxiv_info['categories'].isnull()
drop_indices_list = list(np.where(drop_indices == True)[0])
kept_indices_list = list(np.where(drop_indices == False)[0])
kept_did_list = map(lambda x:x+1, kept_indices_list)
id2arxiv_info = id2arxiv_info[~drop_indices]
unique_did = list(id2arxiv_info.index)
n_docs = np.unique(unique_did).shape[0]

if args.observed_topics:
  document_category_dummies = id2arxiv_info['categories'].str.join(sep='').str.get_dummies(sep=' ')
  category_list = list(document_category_dummies.columns)
  n_categories = len(category_list)
  print 'num categories (k) = {}'.format(n_categories)
  observed_categories = document_category_dummies.as_matrix().astype(np.float32)

  # check if we have zeros in all rows for some docs
  assert len(np.where(~observed_categories.any(axis=1))[0]) == 0

print 'num articles kept is ' + str(len(kept_did_list))
print 'num dropped articles (with no category) is ' + str(len(id2arxiv_info[drop_indices]))

# load num_users
id2arxiv_uid = pd.read_csv(args.user_info_file, header=None, delimiter='\t', names=['uid', 'arxiv_uid'])
n_users = np.unique(id2arxiv_uid.uid).shape[0]

print 'num docs is {}, num users is {}'.format(n_docs, n_users)


print '=> loading data'
train_data, rows, cols = rec_eval.load_data(args.train_file, (n_docs, n_users), args.binarize)

vad_smat, rows_vad, cols_vad = rec_eval.load_data(args.validation_file, (n_docs, n_users), args.binarize)

vad = dict(X_new=vad_smat.data,
           rows_new=rows_vad,
           cols_new=cols_vad)

print '=> running fit'
coder = pmf.PoissonMF(n_components=n_categories, random_state=98765, verbose=True, a=0.1, b=0.1, c=0.1, d=0.1)

coder.fit(train_data, rows, cols, vad, beta=observed_categories)
