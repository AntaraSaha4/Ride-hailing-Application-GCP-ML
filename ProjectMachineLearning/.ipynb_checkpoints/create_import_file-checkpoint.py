import pandas as pd

from google.cloud import storage

BUCKET='ml-fare-prediction-120818' # Assign your bucket name to the variable BUCKET
DELIMITER='/'
# Defining prefixes for different restaurant training sets
PREFIX_ACME='restaurants_train_set/ACME/'
PREFIX_BAMONTE='restaurants_train_set/Bamonte/'
PREFIX_JING_FONG='restaurants_train_set/Jing_Fong/'
PREFIX_KATZ_DELI='restaurants_train_set/Katz_Deli/'
# Creating a list of prefixes for the different restaurant training sets
FOLDERS = [PREFIX_ACME, PREFIX_BAMONTE, PREFIX_JING_FONG, PREFIX_KATZ_DELI]

print(f'BUCKET : {BUCKET}')

#Creating a connection with the GCP Storage and getting the bucket
print('Establishing a connection with GCP Storage')
client = storage.Client()
bucket = client.get_bucket(BUCKET)

print('Retrieving the list of items for creating the import file.')

data = []

# Looping through the prefixes in FOLDERS and retrieving the blobs for each prefix
for folder in FOLDERS :
   blobs = client.list_blobs(BUCKET, prefix=folder, delimiter=DELIMITER)
   # Looping through the blobs and appending the file path and label to the data list
   for blob in blobs:
       label = folder.split('/')[1]
       data.append({
           'FILE_PATH': f'gs://{BUCKET}/{blob.name}',
           'LABEL': label
       })

# Creating a pandas DataFrame from the data list
df = pd.DataFrame(data)

# Exporting the DataFrame to a CSV file named import_file_restaurants.csv without index and header
print('Exporting data from import file to a CSV file')
df.to_csv('import_file_restaurants.csv', index=False, header=False)
