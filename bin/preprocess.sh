#!/bin/bash

DIR=`dirname $0`
MAIN_FOLDER=$($DIR/config.py)
DATA=$MAIN_FOLDER/data
SPLITS=$MAIN_FOLDER/splits
WASSA=$MAIN_FOLDER/wassa_split


## Processing the Affective Text dataset

# Unpack data and embeddings
tar -xzf $DATA/AffectiveText.Semeval.2007.tar.gz -C $DATA
if [ ! -f $DATA/glove.6B.100d.txt ]; then
    unzip $DATA/glove.6B.zip -d $DATA
fi

# Preprocess
AFFTEXT=$DATA/AffectiveText
mkdir -p $AFFTEXT
cat $DATA/AffectiveText.trial/affectivetext_trial.emotions.gold \
    $DATA/AffectiveText.test/affectivetext_test.emotions.gold \
    > $AFFTEXT/emotion_scores.txt
cat $DATA/AffectiveText.trial/affectivetext_trial.xml \
    $DATA/AffectiveText.test/affectivetext_test.xml | \
    grep instance | \
    sed 's|</instance>||' | \
    sed 's|<instance[[:space:]]id="||' | \
    sed 's|">|_|' > $AFFTEXT/instances.txt

# Split the data for 10-fold cross-validation
mkdir -p $SPLITS
DATASIZE=$(wc -l < $AFFTEXT/instances.txt)
FOLDSIZE=$(($DATASIZE / 10))

for FOLD in $(seq 0 9);
do
    mkdir -p $SPLITS/$FOLD
    head -$(($FOLD * $FOLDSIZE)) $AFFTEXT/instances.txt > $SPLITS/$FOLD/instances.train.txt
    tail -$(( ( 10 - $FOLD ) * $FOLDSIZE)) $AFFTEXT/instances.txt | \
	head -$FOLDSIZE > $SPLITS/$FOLD/instances.test.txt
    tail -$(( ( 10 - $FOLD - 1 ) * $FOLDSIZE)) $AFFTEXT/instances.txt >> $SPLITS/$FOLD/instances.train.txt
    head -$(($FOLD * $FOLDSIZE)) $AFFTEXT/emotion_scores.txt > $SPLITS/$FOLD/emotion_scores.train.txt
    tail -$(( ( 10 - $FOLD ) * $FOLDSIZE)) $AFFTEXT/emotion_scores.txt | \
	head -$FOLDSIZE > $SPLITS/$FOLD/emotion_scores.test.txt
    tail -$(( ( 10 - $FOLD - 1 ) * $FOLDSIZE)) $AFFTEXT/emotion_scores.txt >> $SPLITS/$FOLD/emotion_scores.train.txt
done


## Processing the WASSA 2017 dataset

SEED=$DATA/anger-ratings-0to1.train.txt
mkdir -p $WASSA
for EMO in anger fear joy sadness;
do
    cat $DATA/$EMO-ratings-0to1.train.txt $DATA/$EMO-ratings-0to1.dev.gold.txt | \
	cut -f2 > $WASSA/$EMO.instances.train.txt
    cat $DATA/$EMO-ratings-0to1.train.txt $DATA/$EMO-ratings-0to1.dev.gold.txt | \
	cut -f4 > $WASSA/$EMO.scores.train.txt
    cat $DATA/$EMO-ratings-0to1.test.gold.txt | \
	cut -f2 > $WASSA/$EMO.instances.test.txt
    cat $DATA/$EMO-ratings-0to1.test.gold.txt | \
	cut -f4 > $WASSA/$EMO.scores.test.txt
done
