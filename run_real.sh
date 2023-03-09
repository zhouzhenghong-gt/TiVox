#!/bin/bash
echo "begin"

### baseline
## tivox
# CUDA_VISIBLE_DEVICES=0 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug0 --no_reload --save_txt dvg_baseline.txt
# wait

## time_dvg
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --time_mlp --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug1 --no_reload --save_txt dvg_baseline.txt
# wait

## dvg
# CUDA_VISIBLE_DEVICES=0 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug0 --no_reload --save_txt dvg_baseline.txt
# wait

### ablation
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --noallgrid --skipcoarse --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --skipcoarse --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --noallgrid --skipcoarse --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --skipcoarse --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug1 --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=2 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --noallgrid --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug2 --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --gridnum_scale [10000] --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=2 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --noallgrid --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug2 --no_reload --save_txt dvg_baseline.txt
wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --numdensity 4 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --numdensity 2 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --numdensity 6 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --no_color --tivox --numdensity 8 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 220 --no_color --tivox --numdensity 4 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 420 --no_color --tivox --numdensity 4 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 2 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 4 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 6 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 8 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 10 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 20 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 30 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 40 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 50 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 60 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 70 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 80 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 90 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_real.py --render_test --num_voxels 320 --train_num 100 --radius 0.5985 --focal 4086 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

echo "end"