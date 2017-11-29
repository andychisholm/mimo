#!/bin/bash
set -e

DATAPATH=.

#grep '<c74>' $DATAPATH/train.src.txt | head -n 32 > $DATAPATH/valid.sample.txt
#head -n 32 $DATAPATH/valid.src.txt > $DATAPATH/valid.sample.txt

python translate.py \
    -beam_size 5 \
    -model model.chkpt \
    -vocab $DATAPATH/vocab.pt \
    -src $DATAPATH/valid.sample.txt \
    -no_cuda

stty size|if read rows cols; then
        pr -m -t -w $cols $DATAPATH/valid.sample.txt pred.txt
fi;
