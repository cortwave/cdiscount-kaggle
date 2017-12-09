#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=squeezenet1_1 --fold=$i --lr=0.01 --batch-size=1600 --iter-size=4 --epochs=300 --epoch-size=2000000 --validation-size=400000 --optim=sgd;
done
