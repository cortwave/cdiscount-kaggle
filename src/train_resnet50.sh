#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=resnet50 --fold=$i --lr=0.0001 --batch-size=2048 --iter-size=24 --epochs=300 --epoch-size=200000 --validation-size=100000;
done
