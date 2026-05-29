torchrun --standalone --nproc_per_node=gpu train_pretrain.py --use_wandb --use_moe=1 --data_path='../dataset/pretrain_t2t.jsonl'
