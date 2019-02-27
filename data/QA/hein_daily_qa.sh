#!/bin/bash

### count number of files in each category
### count number of new lines in each file

#### change to the correct directory
mydir=~/Documents/MIDS/W266/final-project/data/raw/hein-daily

echo "Speaker Map"
echo "File count:"
find $mydir -name "*SpeakerMap.txt" | wc -l
echo "Line counts:"
find $mydir -name "*SpeakerMap.txt" | xargs wc -l

echo "Description"
echo "File count:"
find $mydir -name "descr*.txt" | wc -l
echo "Line counts:"
find $mydir -name "descr*" | xargs wc -l

echo "By party bigrams"
echo "File count:"
find $mydir -name "byparty*" | wc -l
echo "Line counts:"
find $mydir -name "byparty*" | xargs wc -l

echo "By speaker bigrams"
echo "File count:"
find $mydir -name "byspeaker*" | wc -l
find $mydir -name "byspeaker*" | xargs wc -l

echo "Speeches"
echo "File count:"
find $mydir -name "speeches*" | wc -l
echo "Line counts:"
find $mydir -name "speeches*" | xargs wc -l
