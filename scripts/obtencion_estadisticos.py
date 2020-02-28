import sys
import os
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

data = pd.read_csv("/mnt/pgth04b/DATABASES_CRIS/data_merged.csv")

finalistas_duration = []
finalistas_duration_2014 = []
finalistas_duration_2015 = []
finalistas_duration_2016 = []
finalistas_duration_2017 = []
finalistas_duration_2018 = []
finalistas_duration_2019 = []
no_finalistas_duration = []
no_finalistas_duration_2017 = []
no_finalistas_duration_2018 = []
no_finalistas_duration_2019 = []

labels_list = data['label(0=FINALISTAS/1=NO_FINALISTAS)'].tolist()

for index, finornofin in enumerate(labels_list):
    duration = data.at[index, 'duration']

    if finornofin == '1':
        no_finalistas_duration.append(duration)
        if data.at[index, 'festival_year'] == 2017:
            no_finalistas_duration_2017.append(duration)
        elif data.at[index, 'festival_year'] == 2018:
            no_finalistas_duration_2018.append(duration)
        elif data.at[index, 'festival_year'] == 2019:
            no_finalistas_duration_2019.append(duration)
    elif finornofin == '0':
        finalistas_duration.append(duration)
        if data.at[index, 'festival_year'] == 2014:
            finalistas_duration_2014.append(duration)
        elif data.at[index, 'festival_year'] == 2015:
            finalistas_duration_2015.append(duration)
        elif data.at[index, 'festival_year'] == 2016:
            finalistas_duration_2016.append(duration)
        elif data.at[index, 'festival_year'] == 2017:
            finalistas_duration_2017.append(duration)
        elif data.at[index, 'festival_year'] == 2018:
            finalistas_duration_2018.append(duration)
        elif data.at[index, 'festival_year'] == 2019:
            finalistas_duration_2019.append(duration)

# print(finalistas_duration)
# print(finalistas_duration_2014)
# print(finalistas_duration_2015)
# print(finalistas_duration_2016)
# print(finalistas_duration_2017)
# print(finalistas_duration_2018)
# print(finalistas_duration_2019)

# print(no_finalistas_duration)
# print(no_finalistas_duration_2017)
# print(no_finalistas_duration_2018)
# print(no_finalistas_duration_2019)

num_bins = 10


#### CORTOS FINALISTAS ####

n, bins, patches = plt.hist(finalistas_duration, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Todos los finalistas')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2014, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2014')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2015, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2015')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2016, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2016')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2017, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2017')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2018, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2018')
plt.show()

n, bins, patches = plt.hist(finalistas_duration_2019, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Finalistas de 2019')
plt.show()


#### CORTOS NO FINALISTAS ####

n, bins, patches = plt.hist(no_finalistas_duration, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('Todos los NO finalistas')
plt.show()

n, bins, patches = plt.hist(no_finalistas_duration_2017, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('NO fnalistas de 2017')
plt.show()

n, bins, patches = plt.hist(no_finalistas_duration_2018, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('NO finalistas de 2018')
plt.show()

n, bins, patches = plt.hist(no_finalistas_duration_2019, num_bins, facecolor='red', alpha=0.5)
plt.xlabel('Duración en segundos')
plt.ylabel('Cantidad de videos')
plt.title('NO finalistas de 2019')
plt.show()

#### DIVIDIENDO ENTRE CORTOS Y LARGOS ####

years = ['2014', '2015', '2016', '2017', '2018', '2019']

finalista_corto_2014 = []
finalista_corto_2015 = []
finalista_corto_2016 = []
finalista_corto_2017 = []
finalista_corto_2018 = []
finalista_corto_2019 = []
finalista_largo_2014 = []
finalista_largo_2015 = []
finalista_largo_2016 = []
finalista_largo_2017 = []
finalista_largo_2018 = []
finalista_largo_2019 = []
num_fin_corto_2014 = 0
num_fin_corto_2015 = 0
num_fin_corto_2016 = 0
num_fin_corto_2017 = 0
num_fin_corto_2018 = 0
num_fin_corto_2019 = 0
num_fin_largo_2014 = 0
num_fin_largo_2015 = 0
num_fin_largo_2016 = 0
num_fin_largo_2017 = 0
num_fin_largo_2018 = 0
num_fin_largo_2019 = 0


for y in years:
    if y == '2014':
        for fin in finalistas_duration_2014:
            if fin < 40:
                finalista_corto_2014.append(fin)
                num_fin_corto_2014 += 1
            else:
                finalista_largo_2014.append(fin)
                num_fin_largo_2014 += 1
    elif y == '2015':
        for fin in finalistas_duration_2015:
            if fin < 40:
                finalista_corto_2015.append(fin)
                num_fin_corto_2015 += 1
            else:
                finalista_largo_2015.append(fin)
                num_fin_largo_2015 += 1
    elif y == '2016':
        for fin in finalistas_duration_2016:
            if fin < 40:
                finalista_corto_2016.append(fin)
                num_fin_corto_2016 += 1
            else:
                finalista_largo_2016.append(fin)
                num_fin_largo_2016 += 1
    elif y == '2017':
        for fin in finalistas_duration_2017:
            if fin < 40:
                finalista_corto_2017.append(fin)
                num_fin_corto_2017 += 1
            else:
                finalista_largo_2017.append(fin)
                num_fin_largo_2017 += 1
    elif y == '2018':
        for fin in finalistas_duration_2018:
            if fin < 40:
                finalista_corto_2018.append(fin)
                num_fin_corto_2018 += 1
            else:
                finalista_largo_2018.append(fin)
                num_fin_largo_2018 += 1
    elif y == '2019':
        for fin in finalistas_duration_2019:
            if fin < 40:
                finalista_corto_2019.append(fin)
                num_fin_corto_2019 += 1
            else:
                finalista_largo_2019.append(fin)
                num_fin_largo_2019 += 1


no_finalista_corto_2017 = []
no_finalista_corto_2018 = []
no_finalista_corto_2019 = []
no_finalista_largo_2014 = []
no_finalista_largo_2015 = []
no_finalista_largo_2016 = []
no_finalista_largo_2017 = []
no_finalista_largo_2018 = []
no_finalista_largo_2019 = []
no_num_fin_corto_2014 = 0
no_num_fin_corto_2015 = 0
no_num_fin_corto_2016 = 0
no_num_fin_corto_2017 = 0
no_num_fin_corto_2018 = 0
no_num_fin_corto_2019 = 0
no_num_fin_largo_2014 = 0
no_num_fin_largo_2015 = 0
no_num_fin_largo_2016 = 0
no_num_fin_largo_2017 = 0
no_num_fin_largo_2018 = 0
no_num_fin_largo_2019 = 0

no_years = ['2017', '2018', '2019']

for y in no_years:

    if y == '2017':
        for fin in no_finalistas_duration_2017:
            if fin < 40:
                no_finalista_corto_2017.append(fin)
                no_num_fin_corto_2017 += 1
            else:
                no_finalista_largo_2017.append(fin)
                no_num_fin_largo_2017 += 1
    elif y == '2018':
        for fin in no_finalistas_duration_2018:
            if fin < 40:
                no_finalista_corto_2018.append(fin)
                no_num_fin_corto_2018 += 1
            else:
                no_finalista_largo_2018.append(fin)
                no_num_fin_largo_2018 += 1
    elif y == '2019':
        for fin in no_finalistas_duration_2019:
            if fin < 40:
                no_finalista_corto_2019.append(fin)
                no_num_fin_corto_2019 += 1
            else:
                no_finalista_largo_2019.append(fin)
                no_num_fin_largo_2019 += 1

#### LOS TOTALES ####

no_finalista_largo = []
no_finalista_corto = []
no_num_fin_corto_total = 0
no_num_fin_largo_total = 0

for fin in no_finalistas_duration:
    if fin < 40:
        no_finalista_corto.append(fin)
        no_num_fin_corto_total += 1
    else:
        no_finalista_largo.append(fin)
        no_num_fin_largo_total += 1

finalista_largo = []
finalista_corto = []
num_fin_corto_total = 0
num_fin_largo_total = 0

for fin in finalistas_duration:
    if fin < 40:
        finalista_corto.append(fin)
        num_fin_corto_total += 1
    else:
        finalista_largo.append(fin)
        num_fin_largo_total += 1
#### PLOT DE LOS CORTOS JUNTOS ####



labels = ['2014', '2015', '2016', '2017', '2018', '2019', 'total']

finalistas_cortos = [num_fin_corto_2014, num_fin_corto_2015, num_fin_corto_2016, num_fin_corto_2017, num_fin_corto_2018,
              num_fin_corto_2019, num_fin_corto_total]
no_finalistas_cortos = [no_num_fin_corto_2014, no_num_fin_corto_2015, no_num_fin_corto_2016, no_num_fin_corto_2017,
                 no_num_fin_corto_2018, no_num_fin_corto_2019, no_num_fin_corto_total]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, finalistas_cortos, width, label='Finalistas')
rects2 = ax.bar(x + width/2, no_finalistas_cortos, width, label='No finalistas')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cantidad de videos')
ax.set_title('Videos cortos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()



#### PLOT DE LOS LARGOS JUNTOS ####


finalistas_largos = [num_fin_largo_2014, num_fin_largo_2015, num_fin_largo_2016, num_fin_largo_2017, num_fin_largo_2018,
              num_fin_largo_2019, num_fin_largo_total]
no_finalistas_largos = [no_num_fin_largo_2014, no_num_fin_largo_2015, no_num_fin_largo_2016, no_num_fin_largo_2017,
                 no_num_fin_largo_2018, no_num_fin_largo_2019, no_num_fin_largo_total]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, finalistas_largos, width, label='Finalistas')
rects2 = ax.bar(x + width/2, no_finalistas_largos, width, label='No finalistas')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cantidad de videos')
ax.set_title('Videos largos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()







