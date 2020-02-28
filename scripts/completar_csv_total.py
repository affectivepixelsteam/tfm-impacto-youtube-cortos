import sys
import os
import numpy as np
import pandas as pd

from csv import writer
from csv import reader


def add_column_in_csv(input_file, output_file, transform_row):
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)



data = pd.read_csv("/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_vs_NOFINALISTAS.csv")

data_merged = data

years = ['2014', '2015', '2016']

for y in years:
    folder_path = os.path.join('/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_ORIGINAL/DATABASES', y)
    dirs = os.listdir(folder_path)
    for directorio in dirs:
        path_inside_directorio = os.path.join(folder_path, directorio)
        os.chdir(path_inside_directorio)
        indirs = os.listdir(path_inside_directorio)
        for file in indirs:
            if file.endswith('complete.csv'):
                data_file = file
                output = directorio + '_total.csv'
                header_of_new_col = 'label(0=FINALISTAS/1=NO_FINALISTAS)'
                default_text = 0
                add_column_in_csv(data_file, output,
                                  lambda row, line_num: row.append(header_of_new_col) if line_num == 1 else row.append(
                                      default_text))

