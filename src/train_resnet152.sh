#!/bin/bash
for i in {0..4};
  do python3 model.py train --architecture=resnet152 --fold=$i --lr=0.01 --batch-size=512 --iter-size=8 --epochs=200 --epoch-size=200000 --validation-size=100000;
done
