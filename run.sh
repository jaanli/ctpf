#!/bin/bash
# this script is for debugging and quick runs
dir=/home/statler/lcharlin/arxiv/dat/dataset_toy/
#dir=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012_clean/
python job_handler.py \
  --train_file=${dir}train.tsv \
  --validation_file=${dir}validation.tsv \
  --test_file=${dir}test.tsv \
  --item_info_file=${dir}items_arxiv_info.tsv \
  --user_info_file=${dir}users.tsv \
  --out_dir=/home/waldorf/altosaar/projects/arxiv/fit/ctpf-debug \
  --binarize \
  --observed_topics \
  --debug