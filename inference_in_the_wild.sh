# NOTE: BATCH SIZE CAN ONLY BE SET TO 1!

# python ./scripts/inference.py \
# --images_dir=./experiment/inference_sources  --n_sample=100 --edit_attribute='inversion'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt \
# --gpu "4" \

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='lip'  \
# --save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

python ./scripts/inference_in_the_wild.py \
--images_dir=./experiment/wild_sources  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
--save_dir=./experiment/wild_results --num_workers=2 ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 
