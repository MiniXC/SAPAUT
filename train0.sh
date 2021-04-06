export CUDA_VISIBLE_DEVICES=0
python train.py \
  --run-name no_pauses_tagging \
  --model-name distilroberta-base \
  --weight-decay 0.01 \
  --lr 5e-5 \
  --batch-size 128 \
  --gradient-accumulation 1 \
  --epochs 4 \
  --log-steps 100 \
  --resample None \
  --lookahead 0 4 \
  --max-length 32 \
  --truncate-left \
  --include-pauses \
  --replace-pause ... \
  --pause-threshold 100 \
  --train-dataset mgb-train \
  --validation-dataset mgb-validation \
  --num-proc 8 \
  --dryrun \
  --tagging