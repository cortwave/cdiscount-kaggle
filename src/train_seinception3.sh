#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=seinception --fold=$i --lr=0.001 --batch-size=1024 --iter-size=8 --epochs=10000 --epoch-size=200000 --validation-size=100000 --optim=sgd;
done
