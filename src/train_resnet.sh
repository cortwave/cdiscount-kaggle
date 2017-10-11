#!/bin/bash
for i in {0..4};
  do python3 model.py train --architecture=resnet50 --fold=$i --lr=0.0001 --batch-size=512 --iter-size=4 --epochs=200 --epoch-size=200000 --validation-size=100000;
done
