#!/bin/bash
for i in 1;
  do python3 model.py train --architecture=seresnet101 --fold=$i --lr=0.01 --batch-size=512 --iter-size=8 --epochs=1000 --epoch-size=400000 --validation-size=300000 --optim=sgd --patience=8;
done
