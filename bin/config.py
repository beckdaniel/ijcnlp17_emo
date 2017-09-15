#!/usr/bin/env python

import os

MAIN_FOLDER = os.path.join(os.environ['HOME'], 'ijcnlp17_emo')
EMBS = os.path.join(MAIN_FOLDER, 'data', 'glove.6B.100d.txt')
EMOS = {'anger': 0,
        'disgust': 1,
        'fear': 2,
        'joy': 3,
        'sadness': 4,
        'surprise': 5}

if __name__ == "__main__":
    print(MAIN_FOLDER)
