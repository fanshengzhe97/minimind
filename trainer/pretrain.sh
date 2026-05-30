
torchrun --standalone --nproc_per_node=8 train_pretrain.py \
  --use_wandb \
  --data_path='../dataset/seq_monkey.jsonl' \
  --epochs=1 \
  --num_hidden_layers=16 \
  --hidden_size=1536 \
  --batch_size=128 \
  --accumulation_steps=2 \
  --learning_rate=5e-4 \
  --warmup_ratio=0.01
