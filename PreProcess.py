import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


# Function to standardize float values
def standardize(ip):
    mean = np.mean(ip)
    sd = np.std(ip)
    for i in range(0,len(ip)):
        tempval = float((float(ip[i]) - mean)/sd)
        ip[i] = tempval


# Function to convert str to int
def converttype(ip):
    count = ip.nunique()
    lst = ip.unique()
    for i in range(0,count):
        ip.replace(lst[i-1],i,inplace=True)
    return count


# Function to perform pre processing of data
def preprocessing(reader , write_path):
    reader.replace('?', np.nan, inplace=True)
    reader.replace(' ?', np.nan, inplace=True)
    reader.replace('', float(np.nan), inplace=True)
    reader.dropna(how='any', inplace=True)
    reader = reader.reset_index(drop=True)
    attr = len(reader.columns)
    for i in range(0, attr):
        if isinstance(reader[i][0], float) is True:
            standardize(reader[i])
        elif isinstance(reader[i][0],np.int64) is True:
            standardize(reader[i])
        else:
            cnt = converttype(reader[i])
    reader.to_csv(write_path, index=False)

# Main Function
read_path, write_path = input().split()
reader = pd.read_csv(open(read_path), header=None)
preprocessing(reader , write_path)
print("Preprocessing Completed...")