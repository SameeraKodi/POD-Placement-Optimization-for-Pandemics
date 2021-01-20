import pandas as pd
import numpy as np
import csv
import yaml
import json
import requests


with open('secrets.yaml', 'r') as f:
  # loads contents of secrets.yaml into a python dictionary
  secret_config = yaml.safe_load(f.read())['api keys']

######################### Getting households and population for each ZIP #####################################

# file that has ZIP codes for Allegheny county (obtained from data.gov)
zip_file = "Allegheny_County_Zip_Code_Boundaries.csv"
zip_codes = pd.read_csv(zip_file)
allegheny_zip = zip_codes[zip_codes.COUNTYFIPS == 42003]
zip_list = list(set(list(allegheny_zip.ZIP)))

data = []
households = []

# getting data from ACS api
for z in zip_list:
    
    url = "https://api.census.gov/data/2018/acs/acs5/profile?get=DP02_0001E"\
          "&for=zip%20code%20tabulation%20area:{}&key={}".format(z, secret_config['acs_api'])
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.content.decode('utf-8'))
        data = json.dumps(data).replace('null','""')
        new_data = json.loads(data)
        
        households.append(new_data[1])

    else:
        #print("You got error: {}".format(response.status_code))
        pass
    
df = pd.DataFrame(households, columns = ["households", "zip"])
df.households = df.households.astype("int")

# getting population for computing population density
data = []
population = []

for z in zip_list:
    
    url = "https://api.census.gov/data/2017/acs/acs5?get=B01003_001E"\
          "&for=zip%20code%20tabulation%20area:{}&key={}".format(z, secret_config['acs_api'])
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.content.decode('utf-8'))
        data = json.dumps(data).replace('null','""')
        

        new_data = json.loads(data)
        population.append(new_data[1])

    else:
        # print("You got error: {}".format(response.status_code))
        pass

pop_df = pd.DataFrame(population, columns = ["population", "zip"])
pop_df.population = pop_df.population.astype("int")
df = df.merge(pop_df, on = 'zip')

cols = df.columns.tolist()
cols = ['zip', 'population', 'households']
df = df[cols]
df = df.sort_values(by = 'zip')

# importing area by zip gathered from blog.splitwise.com
den = pd.read_csv("Zipcode-ZCTA-Population-Density-And-Area-Unsorted.csv")
den = den.drop(['2010 Population','Density Per Sq Mile'], axis = 1)
den.columns = ['zip', 'Land-Sq-Mi']
df.zip = df.zip.astype("int")
df = df.merge(den, on = 'zip')
df['pop_den'] = df['population']/df['Land-Sq-Mi']

df.to_csv("zip_level_data.csv")
