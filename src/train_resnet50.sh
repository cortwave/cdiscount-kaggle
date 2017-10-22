#!/bin/bash
for i in 1;
  do python3 model.py train --architecture=resnet50 --fold=$i --lr=0.001 --batch-size=1024 --iter-size=12 --epochs=300 --epoch-size=200000 --validation-size=100000;
done
