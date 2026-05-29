
torchrun --standalone --nproc_per_node=8 train_pretrain.py --use_wandb --data_path='../dataset/pretrain_t2t.jsonl' --epochs=1 --num_hidden_layers=4 --hidden_size=1064 --batch_size=256 --accumulation_steps=1 --learning_rate=5e-4 --warmup_ratio=0.01
