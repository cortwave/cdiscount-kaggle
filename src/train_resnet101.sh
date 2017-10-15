#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=resnet101 --fold=$i --lr=0.001 --batch-size=64 --iter-size=1 --epochs=200 --epoch-size=200000 --validation-size=100000;
done
