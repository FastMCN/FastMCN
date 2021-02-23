#!/bin/bash

# echo "==============================================================================="
# echo "Compression"
# echo "==============================================================================="
# zstd -19 -f ./ncbi_data.csv
# zstd -19 -f ./ncbi_ctd_data.csv
# zstd -19 -f ./cdr_data.csv
# zstd -19 -f ./cdr_ctd_data.csv

echo -e "\n==============================================================================="
echo "Decompression"
echo "==============================================================================="
cd ./proc_data
zstd -df ./ncbi_data.csv.zst
zstd -df ./ncbi_ctd_data.csv.zst
zstd -df ./cdr_data.csv.zst
zstd -df ./cdr_ctd_data.csv.zst

echo -e "\n==============================================================================="
echo "Prepare data"
echo "==============================================================================="
cd ..
python ./utils/prepare_data_use_tokenizer.py ./proc_data/ncbi_ctd_data.csv ./proc_data/ncbi_data.csv ./proc_data/ncbi_ctd 32768
python ./utils/prepare_data_use_tokenizer.py ./proc_data/cdr_ctd_data.csv ./proc_data/cdr_data.csv ./proc_data/cdr_ctd 32768
