#!/bin/bash
for i in 0 1 2 3 4;
  do python3 model.py train --architecture=resnet50_dropout --fold=$i --lr=0.0001 --batch-size=2048 --iter-size=24 --epochs=3000 --epoch-size=1200000 --validation-size=400000 --optim=sgd --patience=5;
done
