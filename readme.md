### Full README coming soon.....











对verl的修改
新建entropy reward manager，使用这个计算reward

修改数据流，在计算old_log_probs的时候不会把对应的entropy pop掉，同时dataproto里面新增logits，内容。

涉及修改的文件: ec_manager.py, main_ppo.py, fsdp_workers.py, dp_actor.py, ray_trainer.py