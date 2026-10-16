import json
import pandas as pd
from tqdm import tqdm

print("Converting Yelp reviews to CSV...")

reviews = []
with open('../../data/yelp/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        data = json.loads(line)
        reviews.append({
            'user': data['user_id'],
            'item': data['business_id'],
            'rating': data['stars'],
            'timestamp': data['date']
        })

df = pd.DataFrame(reviews)
print(f"Total interactions: {len(df)}")

user_counts = df['user'].value_counts()
item_counts = df['item'].value_counts()

df = df[df['user'].isin(user_counts[user_counts >= 20].index)]
df = df[df['item'].isin(item_counts[item_counts >= 20].index)]

print(f"After 20-core filtering: {len(df)} interactions")
print(f"Users: {df['user'].nunique()}, Items: {df['item'].nunique()}")

df.to_csv('yelp_ratings.csv', index=False)
print("Saved as yelp_ratings.csv")