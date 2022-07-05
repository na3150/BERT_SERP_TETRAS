#!/bin/bash
##!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#OUT_DIR=/results/SQuAD

echo "Container nvidia build = " $NVIDIA_BUILD_ID

output_dir="/workspace/bert/results/"$1
echo "out dir is $output_dir"
mkdir -p $output_dir
if [ ! -d "$output_dir" ]; then
  echo "ERROR: non existing $output_dir"
  exit 1
fi

# ↓This for single GPU
python run_bert_fine_tune_cls.py --data_dir='data/bert_demo' \
 	--config_name='pre_trained_model/KoreALBERT/config/config_albert_base_v2.json' \
       --tokenizer_name='pre_trained_model/KoreALBERT/sentencepiece/sp42g.cased.20191125.model' \
       --task_name='serp' \
       --model_type='albert' \
       --model_name_or_path='pre_trained_model/KoreALBERT/base_v2.pt' \
       --output_dir=$output_dir \
       --per_gpu_train_batch_size=8 \
       --per_gpu_eval_batch_size=8 \
       --do_train \
       --do_eval \
       --learning_rate=5e-5 \
       --num_train_epochs=$2.0 \
       --max_seq_length=512 \
       --overwrite_output_dir \
       --overwrite_cache \
       --eval_all_checkpoints \
       --logging_steps=100 \
       --save_steps=200 \
       --pre_conv_label \
       --pre_sample_train_max=-1 \
       --pre_sample_train_min=-1 \
       --pre_sample_test=-1 \
       --pre_sample_by='LATEST' \
       --pre_delete_html \
       --pre_to_lower \
       --pre_delete_spc_chr \
       --pre_delete_stopword \
       --pre_conv_num_to_zero \
       --pre_convert_word \
