import glob

# 获取所有文件的路径
dataset_files = glob.glob("dataset/*.txt")
print(len(dataset_files))

# 要移除的文件名
filename_to_remove = "dataset/last_iteration.txt"

# 检查并移除特定文件
if filename_to_remove in dataset_files:
    dataset_files.remove(filename_to_remove)
print(len(dataset_files))
from datasets import load_dataset
dataset = load_dataset('text', data_files=dataset_files)
print("Available subsets:", dataset.keys())
ds = dataset['train']
df = ds.to_pandas()
df.head()
import pandas as pd

# Let's say df is your DataFrame

# Create a mask for empty rows
mask = df['text'].isna() | (df['text'] == '')

# Apply the mask and get the index of empty rows
empty_rows_index = df[mask].index

# Check if there are any empty rows
if empty_rows_index.empty:
    print('No empty rows')
else:
    print('Empty rows at the following indices:', empty_rows_index)
len(df)
df = df[df['text'] != '']
len(df)
# Create a mask for duplicate rows
duplicate_mask = df.duplicated(subset=['text'], keep='first')

# Apply the mask and get the index of duplicate rows
duplicate_rows_index = df[duplicate_mask].index

# Check if there are any duplicate rows
if duplicate_rows_index.empty:
    print('No duplicate rows')
else:
    print('Duplicate rows at the following indices:', duplicate_rows_index)
df = df.drop_duplicates(subset=['text'], keep='first')
# Create a mask for duplicate rows
duplicate_mask = df.duplicated(subset=['text'], keep='first')

# Apply the mask and get the index of duplicate rows
duplicate_rows_index = df[duplicate_mask].index

# Check if there are any duplicate rows
if duplicate_rows_index.empty:
    print('No duplicate rows')
else:
    print('Duplicate rows at the following indices:', duplicate_rows_index)
len(df)
mask = ~df['text'].str.contains('NOTE_ON')

# Apply the mask and get the index of rows that do not contain 'NOTE_ON'
no_note_on_index = df[mask].index

# Check if there are any rows that do not contain 'NOTE_ON'
if no_note_on_index.empty:
    print('All rows contain "NOTE_ON"')
else:
    print('Rows that do not contain "NOTE_ON" at the following indices:', no_note_on_index)
pd.set_option('display.max_columns', None)  # This ensures that all columns are displayed
pd.set_option('display.width', None)        # This ensures that each line of output will be as wide as necessary
pd.set_option('display.max_colwidth', None) # This ensures that columns will not be truncated
#row = df.loc[185354] # our dataset is far less than 185354
#print(row)
mask = df['text'].str.contains('NOTE_ON')

# Apply the mask and keep only the rows that contain 'NOTE_ON'
df = df[mask]
df.head()
len(df)
df['text'] = df['text'].str.replace(r'\b=4_4\b', '', case=False, regex=True)
df['text'] = df['text'].str.replace('=4_4', '', regex=False)
df.head()
# Create a new column 'num_words' that contains the number of words in each row of the 'text' column
df['num_words'] = df['text'].str.split().str.len()

# Calculate the highest, lowest, and average number of words
highest_num_words = df['num_words'].max()
lowest_num_words = df['num_words'].min()
average_num_words = df['num_words'].mean()

print('Highest number of words:', highest_num_words)
print('Lowest number of words:', lowest_num_words)
print('Average number of words:', average_num_words)
df = df[df['num_words'] >= 300]
# Calculate the highest, lowest, and average number of words
highest_num_words = df['num_words'].max()
lowest_num_words = df['num_words'].min()
average_num_words = df['num_words'].mean()

print('Highest number of words:', highest_num_words)
print('Lowest number of words:', lowest_num_words)
print('Average number of words:', average_num_words)
# Remove the 'num_words' column
df = df.drop(columns=['num_words'])
from datasets import Dataset
# Convert the DataFrame to a Hugging Face dataset
clean_dataset = Dataset.from_pandas(df)
clean_dataset
clean_dataset = clean_dataset.remove_columns("__index_level_0__")
clean_dataset.push_to_hub("Lancelort5786/MAESTRO_MMM")