# Preliminary QA of data

1. number of files/lines:
```
./hein_daily_qa.sh > hein_daily_qa.txt
```

2. count number of speeches
```
grep -nr "|" speeches_* | cut -d : -f 1 | wc -l
```

3. count number of speaker ids
```
./cnt_speaker_id.sh | cut -c 63- > cnt_speaker_id.txt
```

4. get a dictionary of all speakers
```
cat ../raw/hein-daily/*_SpeakerMap.txt | ./speakermap_qa.py > speakermap_qa.txt
```
