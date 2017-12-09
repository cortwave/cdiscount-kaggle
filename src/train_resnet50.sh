#!/bin/bash
for i in 0 1 2 3 4;
  do python3 model.py train --architecture=resnet50 --fold=$i --lr=0.01 --batch-size=128 --iter-size=1 --epochs=3 --epoch-size=1200000 --validation-size=400000 --optim=sgd --patience=2;
  python3 model.py train --architecture=resnet50 --fold=$i --lr=0.01 --batch-size=1024 --iter-size=12 --epochs=3000 --epoch-size=1200000 --validation-size=400000 --optim=sgd --patience=2;
done
