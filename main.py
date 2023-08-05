import pandas as pd

# Load the dataset
data = pd.read_csv("data/language_detection.csv", encoding='latin-1')

# Get the unique languages in the dataset
unique_languages = data["Language"].unique()

print("Languages that can be predicted:")
print(unique_languages)
