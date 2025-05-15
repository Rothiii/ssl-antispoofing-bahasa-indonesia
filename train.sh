#!/bin/bash
echo "🚀 Running training for different algo"
for i in {0..8}
do
    echo "======================================"
    echo "🚀 Running training for --algo=$i"
    echo "======================================"
    
    # your command to run the training script  
    python train-model.py --comment=perspeaker --algo=$i

    if [ $? -ne 0 ]; then
        echo "❌ Training failed at algo=$i. Stopping."
        exit 1
    fi

    echo "✅ Finished training for --algo=$i"
    echo ""
done

echo "🎉 All trainings completed"
