import json

data = json.load(open('plots.json'))

new = []
a = 0
b = 0
c = 0



data = data[::-1]
w = []
while(b < 1000):
    if data[a] is not None:
        if data[a]['title'] != "" or data[a]["plot"] != "" or data[a]['title'] != None or data[a]["plot"] != None:
            w.append(data[a])
            b+= 1
    a += 1



print c
with open('newplots.json', 'w') as outfile:
    json.dump(w, outfile)
