#!/bin/bash

### count number of unique speaker ids in each congress document

#### change to the correct directory
mydir=~/Documents/MIDS/W266/final-project/data/raw/hein-daily

for file in $mydir/*SpeakerMap.txt
do
    echo -n $file " "
    cut -c 1-8 < $file | sort | uniq | wc -l
done
