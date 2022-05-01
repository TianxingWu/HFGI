# NOTE: BATCH SIZE CAN ONLY BE SET TO 1!


# === smile manipulation ===
# multiple thread config: 
# 1) CUDA_VISIBLE_DEVICES
# 2) thread_id

CUDA_VISIBLE_DEVICES=4 \
python ./scripts/inference_fakenews.py \
--image_root=/mnt/lustre/share/rshao/data/FakeNews/VisualNews/origin/ \
--json_dir=/mnt/lustre/share/txwu/data/FakeNews/Ours/metadata/restrict_faces_num/posts_allconfident.json \
--save_dir=/mnt/lustre/share/txwu/data/FakeNews/Ours/images/manipulation/attribute/ \
--edit_attribute='smile' \
--edit_degree=1.5 \
--num_workers=1 \
--ckpt=./checkpoint/ckpt.pt \
--loop_size=256 \
--thread_num=2 \
--thread_id=1 \


# === age manipulation ===
# multiple thread config: 
# 1) CUDA_VISIBLE_DEVICES
# 2) thread_id

# CUDA_VISIBLE_DEVICES=6 \
# python ./scripts/inference_fakenews.py \
# --image_root=/mnt/lustre/share/rshao/data/FakeNews/VisualNews/origin/ \
# --json_dir=/mnt/lustre/share/txwu/data/FakeNews/Ours/metadata/restrict_faces_num/posts_allconfident.json \
# --save_dir=/mnt/lustre/share/txwu/data/FakeNews/Ours/images/manipulation/attribute/ \
# --edit_attribute='age' \
# --edit_degree=3 \
# --num_workers=1 \
# --ckpt=./checkpoint/ckpt.pt \
# --loop_size=256 \
# --thread_num=2 \
# --thread_id=1 \