
# 1 pre
## 1.1 科学上网
    本地将梯子映射到服务器
## 1.2 convert dataset

## 2 caculate norm
安装丢失的包
```shell
apt-get install -y libgl1-mesa-glx
apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk-3-0
apt-get install -y ffmpeg
```
First, romove before dataset
data is in /root/.cache/huggingface/lerobot/luobai/move_banana_to_box_3
try use 
```bash
rm /root/.cache/huggingface/lerobot/luobai/move_banana_to_box_3
```
Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv run scripts/compute_norm_stats.py --config-name pi0_rcvlab_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_rcvlab_low_mem_finetune --exp-name=pi03 --overwrite
```

## 2 caculate norm