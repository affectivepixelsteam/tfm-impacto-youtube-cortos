import pandas as pd
import numpy as np

csv_file = '/mnt/pgth04b/DATABASES_CRIS/FINALISTAS_vs_NOFINALISTAS.csv'

data = pd.read_csv(csv_file)
df = pd.DataFrame(data)

finalistas_cortos_2017 = []
finalistas_cortos_2018 = []
finalistas_cortos_2019 = []

finalistas_largos_2017 = []
finalistas_largos_2018 = []
finalistas_largos_2019 = []

for index, row in df.iterrows():
    if row['label(0=FINALISTAS/1=NO_FINALISTAS)'] == 0:
        if row['festival_year'] == 2017:
            if row['duration'] > 40:
                finalistas_largos_2017.append(row['id'])
            else:
                finalistas_cortos_2017.append(row['id'])

        elif row['festival_year'] == 2018:
            if row['duration'] > 40:
                finalistas_largos_2018.append(row['id'])
            else:
                finalistas_cortos_2018.append(row['id'])
        elif row['festival_year'] == 2019:
            if row['duration'] > 40:
                finalistas_largos_2019.append(row['id'])
            else:
                finalistas_cortos_2019.append(row['id'])

cantidad_cortos_2017 = len(finalistas_cortos_2017)
cantidad_cortos_2018 = len(finalistas_cortos_2018)
cantidad_cortos_2019 = len(finalistas_cortos_2019)

cantidad_largos_2017 = len(finalistas_largos_2017)
cantidad_largos_2018 = len(finalistas_largos_2018)
cantidad_largos_2019 = len(finalistas_largos_2019)

no_finalistas_cortos_2017 = []
no_finalistas_cortos_2018 = []
no_finalistas_cortos_2019 = []

no_finalistas_largos_2017 = []
no_finalistas_largos_2018 = []
no_finalistas_largos_2019 = []

for index, row in df.iterrows():
    if row['label(0=FINALISTAS/1=NO_FINALISTAS)'] == 1:
        if row['festival_year'] == 2017:
            if row['duration'] > 40:
                if len(no_finalistas_largos_2017) < len(finalistas_largos_2017):
                    no_finalistas_largos_2017.append(row['id'])
            else:
                if len(no_finalistas_cortos_2017) < len(finalistas_cortos_2017):
                    no_finalistas_cortos_2017.append(row['id'])

        elif row['festival_year'] == 2018:
            if row['duration'] > 40:
                if len(no_finalistas_largos_2018) < len(finalistas_largos_2018):
                    no_finalistas_largos_2018.append(row['id'])
            else:
                if len(no_finalistas_cortos_2018) < len(finalistas_cortos_2018):
                    no_finalistas_cortos_2018.append(df['id'])
        elif row['festival_year'] == 2019:
            if row['duration'] > 40:
                if len(no_finalistas_largos_2019) < len(finalistas_largos_2019):
                    no_finalistas_largos_2019.append(row['id'])
            else:
                if len(no_finalistas_cortos_2019) < len(finalistas_cortos_2019):
                    no_finalistas_cortos_2019.append(row['id'])

train_finalistas_largos_2017 = int(0.8*len(finalistas_largos_2017))
test_finalistas_largos_2017 = len(finalistas_largos_2017)-train_finalistas_largos_2017
train_finalistas_largos_2018 = int(0.8*len(finalistas_largos_2018))
test_finalistas_largos_2018 = len(finalistas_largos_2018)-train_finalistas_largos_2018
train_finalistas_largos_2019 = int(0.8*len(finalistas_largos_2019))
test_finalistas_largos_2019 = len(finalistas_largos_2019)-train_finalistas_largos_2019

train_test_finalistas_largos_2017 = ['0']*test_finalistas_largos_2017 + ['1']*train_finalistas_largos_2017
train_test_finalistas_largos_2018 = ['0']*test_finalistas_largos_2018 + ['1']*train_finalistas_largos_2018
train_test_finalistas_largos_2019 = ['0']*test_finalistas_largos_2019 + ['1']*train_finalistas_largos_2019

train_finalistas_cortos_2017 = int(0.8*len(finalistas_cortos_2017))
test_finalistas_cortos_2017 = len(finalistas_cortos_2017)-train_finalistas_cortos_2017
train_finalistas_cortos_2018 = int(0.8*len(finalistas_cortos_2018))
test_finalistas_cortos_2018 = len(finalistas_cortos_2018)-train_finalistas_cortos_2018
train_finalistas_cortos_2019 = int(0.8*len(finalistas_cortos_2019))
test_finalistas_cortos_2019 = len(finalistas_cortos_2019)-train_finalistas_cortos_2019

train_test_finalistas_cortos_2017 = ['0']*test_finalistas_cortos_2017 + ['1']*train_finalistas_cortos_2017
train_test_finalistas_cortos_2018 = ['0']*test_finalistas_cortos_2018 + ['1']*train_finalistas_cortos_2018
train_test_finalistas_cortos_2019 = ['0']*test_finalistas_cortos_2019 + ['1']*train_finalistas_cortos_2019

train_no_finalistas_largos_2017 = int(0.8*len(no_finalistas_largos_2017))
test_no_finalistas_largos_2017 = len(no_finalistas_largos_2017)-train_no_finalistas_largos_2017
train_no_finalistas_largos_2018 = int(0.8*len(no_finalistas_largos_2018))
test_no_finalistas_largos_2018 = len(no_finalistas_largos_2018)-train_no_finalistas_largos_2018
train_no_finalistas_largos_2019 = int(0.8*len(no_finalistas_largos_2019))
test_no_finalistas_largos_2019 = len(no_finalistas_largos_2019)-train_no_finalistas_largos_2019

train_test_no_finalistas_largos_2017 = ['0']*test_no_finalistas_largos_2017 + ['1']*train_no_finalistas_largos_2017
train_test_no_finalistas_largos_2018 = ['0']*test_no_finalistas_largos_2018 + ['1']*train_no_finalistas_largos_2018
train_test_no_finalistas_largos_2019 = ['0']*test_no_finalistas_largos_2019 + ['1']*train_no_finalistas_largos_2019

train_no_finalistas_cortos_2017 = int(0.8*len(no_finalistas_cortos_2017))
test_no_finalistas_cortos_2017 = len(no_finalistas_cortos_2017)-train_no_finalistas_cortos_2017
train_no_finalistas_cortos_2018 = int(0.8*len(finalistas_cortos_2018))
test_no_finalistas_cortos_2018 = len(no_finalistas_cortos_2018)-train_no_finalistas_cortos_2018
train_no_finalistas_cortos_2019 = int(0.8*len(no_finalistas_cortos_2019))
test_no_finalistas_cortos_2019 = len(no_finalistas_cortos_2019)-train_no_finalistas_cortos_2019

train_test_no_finalistas_cortos_2017 = ['0']*test_no_finalistas_cortos_2017 + ['1']*train_no_finalistas_cortos_2017
train_test_no_finalistas_cortos_2018 = ['0']*test_no_finalistas_cortos_2018 + ['1']*train_no_finalistas_cortos_2018
train_test_no_finalistas_cortos_2019 = ['0']*test_no_finalistas_cortos_2019 + ['1']*train_no_finalistas_cortos_2019

df_fin_cortos_2017 = pd.DataFrame({'id': finalistas_cortos_2017,
                                   'year': np.full(len(finalistas_cortos_2017), '2017'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_cortos_2017), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_cortos_2017), '0'),
                                   'train (1) o test (0)' : train_test_finalistas_cortos_2017},
                                  index=list(range(0,len(finalistas_cortos_2017))))

count = len(finalistas_cortos_2017)


df_fin_cortos_2018 = pd.DataFrame({'id': finalistas_cortos_2018,
                                   'year': np.full(len(finalistas_cortos_2018), '2018'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_cortos_2018), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_cortos_2018), '0'),
                                   'train (1) o test (0)' : train_test_finalistas_cortos_2018},
                                  index=list(range(len(finalistas_cortos_2017),len(finalistas_cortos_2018) + count)))

count += len(finalistas_cortos_2018)

df_fin_cortos_2019 = pd.DataFrame({'id': finalistas_cortos_2019,
                                   'year': np.full(len(finalistas_cortos_2019), '2019'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_cortos_2019), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_cortos_2019), '0'),
                                   'train (1) o test (0)' : train_test_finalistas_cortos_2019},
                                  index=list(range(count,len(finalistas_cortos_2019) + count)))

count += len(finalistas_cortos_2019)

df_fin_largos_2017 = pd.DataFrame({'id': finalistas_largos_2017,
                                   'year': np.full(len(finalistas_largos_2017), '2017'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_largos_2017), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_largos_2017), '1'),
                                   'train (1) o test (0)' : train_test_finalistas_largos_2017},
                                  index=list(range(count,len(finalistas_largos_2017) + count)))

count += len(finalistas_largos_2017)

df_fin_largos_2018 = pd.DataFrame({'id': finalistas_largos_2018,
                                   'year': np.full(len(finalistas_largos_2018), '2018'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_largos_2018), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_largos_2018), '1'),
                                   'train (1) o test (0)' : train_test_finalistas_largos_2018},
                                  index=list(range(count,len(finalistas_largos_2018) + count)))

count += len(finalistas_largos_2018)

df_fin_largos_2019 = pd.DataFrame({'id': finalistas_largos_2019,
                                   'year': np.full(len(finalistas_largos_2019), '2019'),
                                   'finalista (0) o no finalista (1)': np.full(len(finalistas_largos_2019), '0'),
                                   'largo (1) o corto (0)': np.full(len(finalistas_largos_2019), '1'),
                                   'train (1) o test (0)' : train_test_finalistas_largos_2019},
                                  index=list(range(count,len(finalistas_largos_2019) + count)))

count += len(finalistas_largos_2019)

df_no_fin_cortos_2017 = pd.DataFrame({'id': no_finalistas_cortos_2017,
                                      'year': np.full(len(no_finalistas_cortos_2017), '2017'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_cortos_2017), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_cortos_2017), '0'),
                                      'train (1) o test (0)' : train_test_no_finalistas_cortos_2017},
                                     index=list(range(count,len(no_finalistas_cortos_2017) + count)))

count += len(no_finalistas_cortos_2017)

df_no_fin_cortos_2018 = pd.DataFrame({'id': no_finalistas_cortos_2018,
                                      'year': np.full(len(no_finalistas_cortos_2018), '2018'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_cortos_2018), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_cortos_2018), '0'),
                                      'train (1) o test (0)' : train_test_no_finalistas_cortos_2018},
                                     index=list(range(count,len(no_finalistas_cortos_2018) + count)))

count += len(no_finalistas_cortos_2018)

df_no_fin_cortos_2019 = pd.DataFrame({'id': no_finalistas_cortos_2019,
                                      'year': np.full(len(no_finalistas_cortos_2019), '2019'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_cortos_2019), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_cortos_2019), '0'),
                                      'train (1) o test (0)' : train_test_no_finalistas_cortos_2019},
                                     index=list(range(count,len(no_finalistas_cortos_2019) + count)))

count += len(no_finalistas_cortos_2019)

df_no_fin_largos_2017 = pd.DataFrame({'id': no_finalistas_largos_2017,
                                      'year': np.full(len(no_finalistas_largos_2017), '2017'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_largos_2017), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_largos_2017), '1'),
                                      'train (1) o test (0)' : train_test_no_finalistas_largos_2017},
                                     index=list(range(count,len(no_finalistas_largos_2017) + count)))

count += len(no_finalistas_largos_2017)

df_no_fin_largos_2018 = pd.DataFrame({'id': no_finalistas_largos_2018,
                                      'year': np.full(len(no_finalistas_largos_2018), '2018'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_largos_2018), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_largos_2018), '1'),
                                      'train (1) o test (0)' : train_test_no_finalistas_largos_2018},
                                     index=list(range(count,len(no_finalistas_largos_2018) + count)))

count += len(no_finalistas_largos_2018)

df_no_fin_largos_2019 = pd.DataFrame({'id': no_finalistas_largos_2019,
                                      'year': np.full(len(no_finalistas_largos_2019), '2019'),
                                      'finalista (0) o no finalista (1)': np.full(len(no_finalistas_largos_2019), '1'),
                                      'largo (1) o corto (0)': np.full(len(no_finalistas_largos_2019), '1'),
                                      'train (1) o test (0)' : train_test_no_finalistas_largos_2019},
                                     index=list(range(count,len(no_finalistas_largos_2019) + count)))

count += len(no_finalistas_largos_2019)

frames = [df_fin_cortos_2017, df_fin_cortos_2018, df_fin_cortos_2019, df_fin_largos_2017, df_fin_largos_2018,
          df_fin_largos_2019, df_no_fin_cortos_2017, df_no_fin_cortos_2018, df_no_fin_cortos_2019,
          df_no_fin_largos_2017, df_no_fin_largos_2018, df_no_fin_largos_2019]

total_df = pd.concat(frames)

# train 1 test 0


print(total_df)

total_df.to_csv('/mnt/pgth04b/DATABASES_CRIS/train_test_divition.csv', index = False)