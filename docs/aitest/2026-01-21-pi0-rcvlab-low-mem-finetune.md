apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg
GIT_LFS_SKIP_SMUDGE=1 uv add "lerobot @ git+https://github.com/huggingface/lerobot" --prerelease=allow
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run scripts/compute_norm_stats.py --config-name pi0_rcvlab_low_mem_finetune
