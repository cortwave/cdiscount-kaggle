#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=dpn68 --fold=$i --lr=0.001 --batch-size=1024 --iter-size=16 --epochs=300 --epoch-size=200000 --validation-size=100000 --optim=adam;
done
