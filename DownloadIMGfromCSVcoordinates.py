import requests
import csv
import numpy as np
from pathlib import Path
from decimal import Decimal

Xcoord = np.zeros(3000)
Ycoord = np.zeros(3000)
kmlName = 'Zona Casteldebole.kml'


with open('data/Bologna gps data/' + kmlName) as kml:
    reader = csv.DictReader(kml)  # read rows into a dictionary format
    i = 0
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            if k == ' Xcoord':
                Xcoord[i] = Decimal(v)
            if k == 'Ycoord':
                Ycoord[i] = Decimal(v)
        i += 1

coordNum = i
Xcoord = Xcoord[:coordNum]
Ycoord = Ycoord[:coordNum]
print(Xcoord)
print(Ycoord)

startnsize = 'https://maps.googleapis.com/maps/api/streetview'
size = '?size=300x300'
location = '&location='
heading = '&heading='
pitch = '&pitch=0'
APIkey = '&key=AIzaSyCKBuiVEKcPBJk3h29HQxXPOUwOb59F3Cc'
for i in range(coordNum+1):
    for j in range(4):
        image_url = startnsize + size + location + str(Xcoord[i-1]) + ',' + str(Ycoord[i-1]) \
                    + heading + str(j * 90) + pitch + APIkey
        fileName = 'data/Casteldebole/' + str(Xcoord[i-1]) + 'X_' + str(Ycoord[i-1]) + 'Y_' + str(j*90) + 'deg.jpg'
        if not Path(fileName).exists():
            img_data = requests.get(image_url).content
            with open(fileName, 'wb') as handler:
                handler.write(img_data)
