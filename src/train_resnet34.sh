#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=resnet34 --fold=$i --lr=0.01 --batch-size=128 --iter-size=1 --epochs=10 --epoch-size=200000 --validation-size=100000;
  python3 model.py train --architecture=resnet34 --fold=$i --lr=0.001 --batch-size=128 --iter-size=1 --epochs=10 --epoch-size=200000 --validation-size=100000;
  python3 model.py train --architecture=resnet34 --fold=$i --lr=0.0001 --batch-size=1024 --iter-size=12 --epochs=10 --epoch-size=200000 --validation-size=100000;
done
