#!/bin/zsh
for n in 16 32 64 128 256 512 1024
do
    for dropout in 0 0.2 0.4 0.5 0.6
    do
        python3 ./gru/main.py $n $dropout > log_` date "+%Y%m%d_%H%M%S"`.txt
    done
done
# $n $dropout
