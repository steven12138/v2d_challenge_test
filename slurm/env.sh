# Sourced by the sbatch scripts: activate a uv env and keep caches on scratch.
#   source slurm/env.sh rg      # prep, retarget
#   source slurm/env.sh flash   # train, evaluate

V2D=/scratch/user/the/v2d
ENV=$V2D/envs/$1
export PATH="$ENV/bin:$PATH"
export WARP_CACHE_PATH="$V2D/t2/cache/warp"
export XDG_CACHE_HOME="$V2D/t2/cache/xdg"
export HF_HOME="$V2D/.hf_cache"
export PYTHONUNBUFFERED=1

# onnxruntime-gpu (SONIC encoder/decoder) needs the CUDA 13 + cuDNN 9 libraries. The official image has
# them system-wide; here they come from the NVIDIA pip wheels inside the env.
NV=$ENV/lib/python3.11/site-packages/nvidia
if [[ -d $NV/cu13/lib ]]; then
    export LD_LIBRARY_PATH="$NV/cu13/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
