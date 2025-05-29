#!/bin/bash
echo "🚀 Running check eer for different algo"
for i in 0 1 2 3 4 5 6 7 8
do
    echo "======================================"
    echo "🚀 Running check eer for --algo=$i"
    echo "======================================"
    
    # your command to run the training script  
    python check_eer_perfolder.py scores/8detikssl/algo$i dataset/LA/ASVspoof_LA_cm_protocols

    if [ $? -ne 0 ]; then
        echo "❌ Training failed at algo=$i. Stopping."
        exit 1
    fi

    echo "✅ Finished check eer for --algo=$i"
    echo ""
done

echo "🎉 All check eer perfolder completed"
