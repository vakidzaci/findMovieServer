import json

href = json.load(open('mlproject/movies.json'))
plots = json.load(open('mlproject/plots.json'))



# a = 0
# b = []
# for d in data:
#     if d != None:
#         if 'title' in d and 'href' in d:
#             if d['title']==None or d['href']==None:
#                 a += 1
#             else:
#                 b.append(d)
#         else:
#             a+=1
#     else:
#         a +=1
# print a
#
#
#
# with open('mlproject/movies.json', 'w') as outfile:
#     json.dump(b, outfile)



for h in href:
    for p in plots:
        if(h['title'] == p['title']):
            p['href'] = h['href']




with open('mlproject/plots.json', 'w') as outfile:
    json.dump(plots, outfile)
