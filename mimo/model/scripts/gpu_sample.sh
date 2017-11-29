#!/bin/bash
set -e

DATAPATH=dataset
SAMPLE_SRC_PATH=$DATAPATH/valid.src.sample.txt
SAMPLE_TGT_PATH=$DATAPATH/valid.tgt.sample.txt

grep '<c74>' $DATAPATH/valid.src.txt | head -n 32 > $SAMPLE_SRC_PATH
grep '<c74>' $DATAPATH/valid.tgt.txt | head -n 32 > $SAMPLE_TGT_PATH

#head -n 512 $DATAPATH/valid.src.txt > $SAMPLE_SRC_PATH
#head -n 512 $DATAPATH/valid.tgt.txt > $SAMPLE_TGT_PATH

#head -n 32 $DATAPATH/valid.src.txt > $DATAPATH/valid.sample.txt

python translate.py \
    -beam_size 5 \
    -n_best 2 \
    -model trained.chkpt \
    -vocab $DATAPATH/dataset.pt \
    -src $SAMPLE_SRC_PATH \
    -output $DATAPATH/system.txt

awk '{for(i=0;i<2;i++)print}' $SAMPLE_SRC_PATH > $SAMPLE_SRC_PATH.tmp
mv $SAMPLE_SRC_PATH.tmp $SAMPLE_SRC_PATH

stty size|if read rows cols; then
        pr -m -t -w $cols $SAMPLE_SRC_PATH $DATAPATH/system.txt
fi;
