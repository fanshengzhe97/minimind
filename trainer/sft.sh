
torchrun --standalone --nproc_per_node=8 train_full_sft.py \
  --use_wandb \
  --data_path='../dataset/sft_t2t.jsonl' \
  --epochs=2 \
  --num_hidden_layers=8 \
  --hidden_size=768 \
  --use_moe=0 \
  --batch_size=128 \
  --accumulation_steps=1 \
  --learning_rate=1e-5 \
  --save_weight=full_sft \
  --from_weight='MiniMind-Pretrain-DSpretrain_t2t-L8-H768-S340-MoE0K0-BS256-GA1-LR0.0005-Ep1-P63p9M-A63p9M' \
  --from_resume=0
