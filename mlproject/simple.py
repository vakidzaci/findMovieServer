import json

def get(_id):
    data = json.load(open('mlproject/newplots.json'))
    return data[_id]
