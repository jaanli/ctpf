#!/bin/bash
# this script is for debugging and quick runs

python job_handler.py \
  --train_file=/home/statler/lcharlin/arxiv/dat/dataset_toy/train.tsv \
  --validation_file=/home/statler/lcharlin/arxiv/dat/dataset_toy/validation.tsv \
  --test_file=/home/statler/lcharlin/arxiv/dat/dataset_toy/test.tsv \
  --item_info_file=/home/statler/lcharlin/arxiv/dat/dataset_toy/items_arxiv_info.tsv \
  --user_info_file=/home/statler/lcharlin/arxiv/dat/dataset_toy/users.tsv \
  --out_dir=/home/waldorf/altosaar/projects/arxiv/fit/ctpf-debug \
  --binarize \
  --observed_topics \
  --debug