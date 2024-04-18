import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob  # Sentiment Analysis Library

# Load the dataset (replace 'google_play_store.csv' with your filename)
data = pd.read_csv("google_play_store.csv")

# Data Cleaning
def clean_data(data):
  # Handle missing values (replace with your strategy)
  data.dropna(subset=['Rating', 'Reviews'], inplace=True)

  # Convert Installs to integer
  data['Installs'] = data['Installs'].str.replace(',', '').astype(int)

  # Convert Price to float (if applicable)
  try:
    data['Price'] = data['Price'].str.replace('$', '').astype(float)
  except ValueError:
    pass  # Handle non-numeric pricing

  return data

data = clean_data(data.copy())  # Clean a copy to avoid modifying original data

# Category Exploration
category_counts = data['Category'].value_counts()

# Plot a bar chart for category distribution
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title("Distribution of Apps by Category")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Metrics Analysis
average_rating = data['Rating'].mean()
print(f"Average App Rating: {average_rating:.2f}")

top_rated_apps = data.nlargest(10, 'Rating')
print("\nTop 10 Rated Apps:")
print(top_rated_apps[['App Name', 'Rating']])

largest_apps = data.nlargest(10, 'Size')  # Explore largest apps by size (in MB)
print("\nTop 10 Largest Apps (Size in MB):")
print(largest_apps[['App Name', 'Size']])

# Sentiment Analysis (basic)
def analyze_sentiment(text):
  blob = TextBlob(text)
  return blob.sentiment.polarity

data['Sentiment'] = data['Reviews'].apply(analyze_sentiment)

average_sentiment = data['Sentiment'].mean()
print(f"\nAverage Sentiment Score (positive: > 0, negative: < 0): {average_sentiment:.2f}")

# Filter reviews by sentiment
positive_reviews = data[data['Sentiment'] > 0]['Reviews']
negative_reviews = data[data['Sentiment'] < 0]['Reviews']

# Print a few examples (adjust count)
print("\nPositive Review Examples:")
for review in positive_reviews.head(2):
  print(review)

print("\nNegative Review Examples:")
for review in negative_reviews.head(2):
  print(review)

# Explore correlations (replace with your desired correlations)
correlation = data['Rating'].corr(data['Installs'])
print(f"\nCorrelation between Rating and Installs: {correlation:.2f}")

# Price Distribution by Category (example)
data_by_category = data.groupby('Category')
price_by_category = data_by_category['Price'].describe()

print(f"\nPrice Distribution by Category (descriptive statistics):")
print(price_by_category)

