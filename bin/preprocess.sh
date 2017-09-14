#!/bin/bash

DIR=`dirname $0`
#source $DIR/config.sh
MAIN_FOLDER=$($DIR/config.py)
DATA=$MAIN_FOLDER/data
SPLITS=$MAIN_FOLDER/splits

# Unpack data and embeddings
tar -xzf $DATA/AffectiveText.Semeval.2007.tar.gz -C $DATA
if [ ! -f $DATA/glove.6B.100d.txt ]; then
    unzip $DATA/glove.6B.zip -d $DATA
fi

# Preprocess the data
cat $DATA/AffectiveText.trial/affectivetext_trial.emotions.gold \
    $DATA/AffectiveText.test/affectivetext_test.emotions.gold \
    > $DATA/emotion_scores.txt
cat $DATA/AffectiveText.trial/affectivetext_trial.xml \
    $DATA/AffectiveText.test/affectivetext_test.xml | \
    grep instance | \
    sed 's|</instance>||' | \
    sed 's|<instance[[:space:]]id="||' | \
    sed 's|">|_|' > $DATA/instances.txt

# Split the data for 10-fold cross-validation
mkdir -p $SPLITS
DATASIZE=$(wc -l < $DATA/instances.txt)
FOLDSIZE=$(($DATASIZE / 10))

for FOLD in $(seq 0 9);
do
    mkdir -p $SPLITS/$FOLD
    head -$(($FOLD * $FOLDSIZE)) $DATA/instances.txt > $SPLITS/$FOLD/instances.train.txt
    tail -$(( ( 10 - $FOLD ) * $FOLDSIZE)) $DATA/instances.txt | \
	head -$FOLDSIZE > $SPLITS/$FOLD/instances.test.txt
    tail -$(( ( 10 - $FOLD - 1 ) * $FOLDSIZE)) $DATA/instances.txt >> $SPLITS/$FOLD/instances.train.txt
done
