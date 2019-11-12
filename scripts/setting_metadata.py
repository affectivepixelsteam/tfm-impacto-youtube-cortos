#!/usr/bin/env python3


import pandas as pd
import json
import os
from pandas.io.json import json_normalize


os.chdir('/home/aitorgalan/Escritorio/tfm-impacto-youtube-cortos/DATABASES/2014/_36smCw945g')
json_file = '_36smCw945g.info.json'
with open(json_file) as json_data:
    data = json.load(json_data)

json_normalize(data)

w = 34

dataframe = pd.DataFrame.from_dict(data)
# pd.read_json(json_file)
