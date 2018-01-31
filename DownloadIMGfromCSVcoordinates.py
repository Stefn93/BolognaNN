import requests
import sys, csv
from collections import defaultdict

Xcoord = defaultdict(list) # each value in each column is appended to a list
Ycoord = defaultdict(list)
kmlName = 'Zona Centro 1.kml'


with open(kmlName) as kml:
    reader = csv.DictReader(kml)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            if k == 0:
                Xcoord.append(v)  # append the value into the appropriate list
                                      # based on column name k


#img_data = requests.get(image_url).content
#with open('image_name.jpg', 'wb') as handler:
#    handler.write(img_data)