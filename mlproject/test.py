import json
import sys


data = json.load(open('newplots.json'))


# print data[137]['plot']
#
for i in range(len(data)):
    print "%i : %s"%(i,data[i]['title'])
