#!/usr/bin/env python

# aggregate speakers across all congresses and output them into a file

import sys
from collections import defaultdict

speakermap = defaultdict(list)
for line in sys.stdin:
    line = line.strip().split("|")
    if line[0] != 'speakerid':
        if speakermap[line[0]]:
            speakermap[line[0]][-1] += 1
        else:
            if len(line[0]) == 8:     
                congress = line[0][:2]
            else:
                congress = line[0][:3]
            speakermap[line[0]] = [congress, line[2]+', '+line[3], line[4], line[5], line[6], line[7], 1]                
                
print(f"{len(list(speakermap.keys()))} congresspeople were written to the output file")

with open("full_speakermap.txt", "w") as f:
    for person in speakermap:
        speakermap[person][6] = str(speakermap[person][6])
        payload = '\t'.join(speakermap[person])
        f.write(person+"\t"+payload+"\n")