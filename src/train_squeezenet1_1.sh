#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=squeezenet1_1 --fold=$i --lr=0.001 --batch-size=256 --iter-size=4 --epochs=300 --epoch-size=200000 --validation-size=100000 --optim=adam;
done
