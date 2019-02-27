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
    fullname = line[2] + ', ' + line[3]
    # speechid lengths differ by congress number
    if len(line[0]) == 8:
      # congresses 97-99
      congress = line[0][:2]
    else:
      # congresses 100+
      congress = line[0][:3]
    # if a record for the congressperson exists
    if len(speakermap[fullname]):
      # and this speech is for a different congress
      if congress not in speakermap[fullname][0]:
        # add congress number to the first value list
        speakermap[fullname][0].append(congress)
        # add demo info to the second value list
        # in case they changes chamber or state
        speakermap[fullname][1].append(line[4:7])
    else:
      # otherwise, create a record for the speaker
      speakermap[fullname] = [[congress]]
      speakermap[fullname].append([line[4:7]])

# print one line for each speaker
for person in speakermap:
  # the their name, list of congresses, first demo record (for brevity)
  print(f"{person}, {' '.join(speakermap[person][0])}, {', '.join(speakermap[person][1][0])}")
#  print(f"{person}, {speakermap[person][0]}, {speakermap[person][1][0]}")
