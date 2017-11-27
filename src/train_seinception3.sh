#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=seinception --fold=$i --lr=0.01 --batch-size=512 --iter-size=4 --epochs=10000 --epoch-size=400000 --validation-size=300000 --optim=sgd --ignore-prev-best-loss --patience=10;
done
