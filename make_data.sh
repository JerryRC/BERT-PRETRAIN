python create_pretraining_data.py\
    --input_file=./bert_pretrain_data/shui5_2.txt\
    --output_file=shui5_2\
    --max_seq_length=128\
    --max_predictions_per_seq=20\
    --masked_lm_prob=0.15\
    --random_seed=519\
    --dupe_factor=5