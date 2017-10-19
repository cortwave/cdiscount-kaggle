#!/bin/bash
for i in 0;
  do python3 model.py train --architecture=seresnet50 --fold=$i --lr=0.01 --batch-size=80 --iter-size=1 --epochs=5 --epoch-size=200000 --validation-size=100000;
  python3 model.py train --architecture=seresnet50 --fold=$i --lr=0.01 --batch-size=80 --iter-size=1 --epochs=5 --epoch-size=200000 --validation-size=100000;
  python3 model.py train --architecture=seresnet50 --fold=$i --lr=0.001 --batch-size=1024 --iter-size=16 --epochs=30 --epoch-size=200000 --validation-size=100000;
  python3 model.py train --architecture=seresnet50 --fold=$i --lr=0.001 --batch-size=2048 --iter-size=32 --epochs=300 --epoch-size=200000 --validation-size=100000;
done
