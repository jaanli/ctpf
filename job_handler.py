# this script handles loading, preprocessing of data and launching experiments for:
# vanilla PF, hierarchical PF, user-artist-song PF / observed CTPF
# input: command line arguments specifying train/validation/test files and the algorithm to use
# output: held-out evaluation metrics, training log-likelihood file, saved user preferences / epsilons

import argparse
import sys

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

args = parser.parse_args()

print 'train file is {}'.format(args.train_file)
