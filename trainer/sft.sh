
torchrun --standalone --nproc_per_node=8 train_full_sft.py \
  --use_wandb \
  --data_path='../dataset/sft_t2t.jsonl' \
  --epochs=2 \
  --num_hidden_layers=8 \
  --hidden_size=768 \
  --use_moe=1 \
  --batch_size=128 \
  --accumulation_steps=1 \
  --learning_rate=1e-5 \
  --save_weight=full_sft \
  --from_weight='MiniMind-Pretrain-DSpretrain_t2t-L8-H768-S340-MoE4K1-BS32-GA8-LR0.0005-Ep2-P198p4M-A63p9M' \
  --from_resume=0 \
  --max_seq_len=1024
