# Preliminary QA of data

1. number of files/lines:
```
./hein_daily_qa.sh > hein_daily_qa.txt
```

2. count number of speeches
```
grep -nr "," speeches_* , cut -d : -f 1 , wc -l
```

3. count number of speaker ids
```
./cnt_speaker_id.sh , cut -c 63- > cnt_speaker_id.txt
```

4. get a dictionary of all speakers
```
cat ../raw/hein-daily/*_SpeakerMap.txt , ./speakermap_qa.py
```

5. get a dictionary of all speakers by congress
```
cat ../raw/hein-daily/*_SpeakerMap.txt , ./full_speakermap.py
```

# Created files

- `cnt_speaker_id.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/cnt_speaker_id.sh)
    + Content: Count of speakerids in each speakermap file
    + Row count: 18 (number of congresses)
    + Header: False
    + Columns: [filename, count]
    + Separator: NA
- `hein_daily_qa.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/hein_daily_qa.sh)
    + Content: File and line counts of all files in the raw directory
    + Row count: NA
    + Header: False
    + Columns: Unstructured
    + Separator: NA
- `speakermap_qa.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/speakermap_qa.py)
    + Content: List of unique names for congress people
    + Row count: 1,791 (number of unique names for congress people in 18 congresses that have speech records)
    + Header: False
    + Columns: [Full Name, State, List of congresses, Chamber, State, Gender, Party]
    + Separator: tab
- `congresspeople_age.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/collecting_demo.ipynb)
    + Content: Birth year of each congress person
    + Row count: 1,807 (number of congress people in 18 congresses from congress.gov)
    + Header: False
    + Columns: [Full name, Birth year]
    + Separator: tab
- `congresspeople_id.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/collecting_demo.ipynb)
    + Content: Unique congress ID from [CONGRESS.GOV](https://www.congress.gov/members)
    + Row count: 1,807 (number of congress people in 18 congresses from congress.gov)
    + Header: False
    + Columns: [Full name, ID]
    + Separator: tab
- `congresspeople_demo.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/speakermap_qa.ipynb)
    + Content: All gender info (including ethnicity and age) for all congress people
    + Row count: 1,801 (number of congress people in 18 congresses from congress.gov that have speech records)
    + Header: True
    + Columns: [Full Name, First Name, Last Name, First Last, List of Congresses, Chamber, State, Gender, Party, Ethnicity, BirthYear, Congress Name,  Alternative name (Wikipedia), Congress ID]
    + Separator: pipe
- `ethicity_aapia.txt`, `ethicity_black.txt`, `ethicity_hispanic.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/collecting_demo.ipynb)
    + Content: list of people with years in congress for each ethnicity (race)
    + Row counts: 74, 155, 141 (number of congress people of color on Wikipedia)
    + Header: False
    + Columns: [Full Name, Year start in congress, Year end in congress]
    + Separator: tab
- `full_speakermap.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/full_speakermap.py)
    + Content: aggregated speakermap files
    + Row count: 9,845 (full name + congress number)
    + Header: False
    + Columns: [speaker id, Congress, Full Name, Chamber, State, Gender, Party, Speech Count]
    + Separator: tab
- `full_speakermap_demo.txt`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/EDA/EDA_Yulia_0309_speakermap.ipynb)
    + Content: aggregated speakermap files with demo information and target variables
    + Row count: 9,845 (full name + congress number)
    + Header: True
    + Columns: [speakerid, Congress, Full Name, Chamber, State, Gender, Party, SpeachCount, List of Congresses, Ethnicity, BirthYear, CongressYear, Age,  Age_lt_med, NonWhite, Female, Age_med]
    + Separator: pipe
- `full_descr.zip`
    + [source](https://github.com/YuliaZamriy/W266-Final-Project/blob/master/data/QA/speech_descr_qa.ipynb)
    + Content: concatenated *descr* files with target variables
    + Row count: 2,585,807 (number of speeches with speaker ids for 18 congresses)
    + Header: True
    + Columns: [speech_id, date, char_count, word_count, speakerid, party,  Congress, Ethnicity, Age, Age_lt_med, NonWhite, Female]
    + Separator: pipe


