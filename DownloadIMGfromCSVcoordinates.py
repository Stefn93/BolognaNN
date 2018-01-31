import requests
import sys, csv
from collections import defaultdict
import numpy as np
from decimal import Decimal

Xcoord = np.zeros(3000)
Ycoord = np.zeros(3000)
kmlName = 'Zona Centro.kml'


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

Xcoord = Xcoord[:i]
Ycoord = Ycoord[:i]
print(Xcoord)
print(Ycoord)


#img_data = requests.get(image_url).content
#with open('image_name.jpg', 'wb') as handler:
#    handler.write(img_data)