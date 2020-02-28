import json
files = ['a', 'b']
data = []

for f in files:
    prediction = ['1', '2', '3']
    classes = ['f', 'g', 'h']
    top3 = [0,1,2]
    item = {'id': f}
    for i in top3:
        item[classes[i]] = prediction[i]

    data.append(item)
jsonData = json.dmps(data)
print(jsonData)

