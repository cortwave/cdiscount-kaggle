#!/bin/bash
for i in 1;
  do python3 model.py train --architecture=resnet50 --fold=$i --lr=0.01 --batch-size=1024 --iter-size=12 --epochs=3000 --epoch-size=400000 --validation-size=200000 --optim=sgd --patience=2;
done
