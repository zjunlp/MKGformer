python main.py --gpus "0," --max_epochs=30  --num_workers=32 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --bce 0 \
   --model_class UnimoKGC \
   --batch_size 128 \
   --pretrain 1 \
   --check_val_every_n_epoch 10 \
   --data_dir dataset/WN18 \
   --task_name wn18 \
   --overwrite_cache \
   --eval_batch_size 256 \
   --precision 16 \
   --max_seq_length 32 \
   --lr 7e-5

