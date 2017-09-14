#!/bin/bash

DIR=`dirname $0`
source $DIR/config.sh
DATA=$MAIN_FOLDER/data

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
