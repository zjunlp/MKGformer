python main.py --gpus "0," --max_epochs=31  --num_workers=8 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --bce 1 \
   --model_class UnimoKGC \
   --checkpoint your_pretrained_model_path \
   --batch_size 128 \
   --pretrain 0 \
   --check_val_every_n_epoch 3 \
   --data_dir dataset/WN18 \
   --task_name wn18 \
   --overwrite_cache \
   --eval_batch_size 128 \
   --precision 16 \
   --max_seq_length 32 \
   --lr 5e-5

