#!/bin/bash
for i in {0..4};
  do python3 model.py train --architecture=resnet152 --fold=$i --lr=0.0001 --batch-size=16 --epochs=100 --epoch-size=100000 --validation-size=100000;
done
