#!/bin/bash
echo "begin"

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --skipcoarse --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --numdensity 2 --skipcoarse --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --numdensity 2 --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 50 --skipcoarse --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait
CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
wait

CUDA_VISIBLE_DEVICES=1 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt --idea

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 2 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 4 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 6 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 8 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 10 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 20 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 30 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 40 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 60 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 70 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 80 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 90 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --gridnum_scale [10000] --train_num 100 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 2 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 4 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 6 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 8 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 10 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 20 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 30 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 40 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 60 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 70 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 80 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 90 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --skipcoarse --no_color --tivox --gridnum_scale [10000] --train_num 100 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 2 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 4 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 6 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 8 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 10 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 20 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 30 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 40 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 60 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 70 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 80 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 90 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --tivox --train_num 100 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 2 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 4 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 6 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 8 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 10 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 20 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 30 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 40 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 60 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 70 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 80 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 90 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --no_color --train_num 100 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 2 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 4 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 6 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 8 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 10 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 20 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 30 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 40 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 50 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 60 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 70 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 80 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 90 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait
# CUDA_VISIBLE_DEVICES=3 python run.py --config configs/dvg-dsa/dsa_idea.py --render_test --num_voxels 470 --train_num 100 --radius 0.5985 --focal 2430 --size 1024 --expname debug --no_reload --save_txt dvg_baseline.txt
# wait

echo "end"