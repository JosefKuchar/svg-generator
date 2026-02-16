CUDA_VISIBLE_DEVICES=3 uv run python precompute_text_embeddings.py \
  --dataset JosefKuchar/bezier-dataset \
  --splits train,valid,test \
  --batch-size 128 \
  --max-length 128 \
  --dtype float16 \
  --output bezier_dataset_with_text_embeddings
