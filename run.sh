#!/bin/bash
# this script is for debugging and quick runs

python job_handler.py \
  --train_file=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012/train.tsv \
  --validation_file=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012/validation.tsv \
  --test_file=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012/test.tsv \
  --item_info_file=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012/items_arxiv_info.tsv \
  --user_info_file=/home/statler/lcharlin/arxiv/dat/dataset_2003-2012/users.tsv \
  --out_dir=/home/waldorf/altosaar/projects/arxiv/fit/ctpf-stagewise-heldout \
  --binarize