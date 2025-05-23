#!/bin/bash
set -e

# activate anaconda environment
# source ~/anaconda3/etc/profile.d/conda.sh
conda activate SSL_Spoofing

echo "Start training model"
mkdir -p scores/8detikaudio

for i in {0..8}
do
    echo "🚀 Running eval for --algo=$i"
    mkdir -p "scores/8detikaudio/algo${i}"

    python train-model.py \
        --track=LA \
        --is_eval \
        --eval \
        --models_folder="models/weighted_CCE_20_10_1e-06_${i}_perspeaker/" \
        --eval_output="scores/8detikaudio/algo${i}"

    echo "✅ Finish eval for --algo=$i"
done

echo "🎉 All evals completed"
