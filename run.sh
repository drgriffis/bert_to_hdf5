#!/bin/bash
#
# Demo script, illustrating steps to get HDF5-format embeddings
# from BERT, with aligned tokenization.
#

set -e

PYTHON=python
MODEL=BERT-Base
BERT=${HOME}/bert
GPU=0

## Fetch BERT files from per-model specifications in config.ini
VOCABFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} VocabFile")
CONFIGFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} ConfigFile")
CKPTFILE=$(${PYTHON} -m cli_configparser.read_setting -c config.ini BERT "${MODEL} CkptFile")

INPUT=demo.txt
PRE_TOKENIZED=demo.pre_tokenized_for_BERT.${MODEL}
JSON=demo.json1
HDF5=demo.hdf5
SEQ_LENGTH=128

export PYTHONPATH=${PYTHONPATH}:${BERT}

if [ ! -e "${PRE_TOKENIZED}.tokens" ]; then
    ${PYTHON} -m pre_tokenize_for_BERT \
        -i ${INPUT} \
        -o ${PRE_TOKENIZED} \
        -s ${SEQ_LENGTH} \
        --overlap 0.5 \
        -v ${VOCABFILE}
fi

if [ ! -e "${JSON}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU}
    ${PYTHON} -m extract_features_pretokenized \
        --input_file ${PRE_TOKENIZED}.tokens \
        --output_file ${JSON} \
        --vocab_file ${VOCABFILE} \
        --bert_config_file ${CONFIGFILE} \
        --init_checkpoint ${CKPTFILE} \
        --layers -1,-2,-3 \
        --max_seq_length ${SEQ_LENGTH} \
        --batch_size 8
fi

if [ ! -e "${HDF5}" ]; then
    ${PYTHON} -m recombine_BERT_embeddings \
        --bert-output ${JSON} \
        --overlaps ${PRE_TOKENIZED}.overlaps \
        -o ${HDF5} \
        --tokenized ${HDF5}.aligned_tokens.txt \
        -l ${HDF5}.log
fi
