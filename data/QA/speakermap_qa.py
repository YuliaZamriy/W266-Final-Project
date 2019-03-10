#!/usr/bin/env python

# inspiration: https://stackabuse.com/read-a-file-line-by-line-in-python/
# aggregate speakers across all congresses and output them into a file

import sys
from collections import defaultdict

speakermap = defaultdict(list)
for line in sys.stdin:
  line = line.strip().split("|")
  # skip the header line
  if line[0] != 'speakerid':
    # concatenate last and first names
    fullname = line[2]+', '+line[3]
    # get chamber, state, gender, party
    payload = line[5]
    # put it all in one key
    full_key = fullname+"\t"+payload
    # speechid lengths differ by congress number
    if len(line[0]) == 8:
      # congresses 97-99
      congress = line[0][:2]
    else:
      # congresses 100+
      congress = line[0][:3]
    # if a record for the congressperson exists
    if len(speakermap[full_key]):
      # and this speech is for a different congress
      if congress not in speakermap[full_key][0]:
        # add congress number to the first value list
        speakermap[full_key][0].append(congress)
        speakermap[full_key][1].append(line[4:8])
    else:
      # otherwise, create a record for the speaker
      speakermap[full_key] = [[congress]]
      speakermap[full_key].append([line[4:8]])

print(f"{len(list(speakermap.keys()))} congresspeople were written to the output file")
        
with open("speakermap_qa.txt", "w") as f:
    for person in speakermap:
        congresses = ' '.join(speakermap[person][0])
        payload = ' '.join(speakermap[person][1][0])
        f.write(person+"\t"+congresses+"\t"+payload+"\n")