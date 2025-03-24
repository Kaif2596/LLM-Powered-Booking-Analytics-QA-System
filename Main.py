import pandas as pd

# Load the dataset (assuming it’s saved as 'hotel_bookings.csv')
df = pd.read_csv('D:\Assignment\hotel_bookings.csv')

# Check for missing values
print(df.isnull().sum())

# Handle missing values (example: fill 'country' with 'Unknown', drop rows with missing 'adr')
df['country'].fillna('Unknown', inplace=True)
df.dropna(subset=['adr'], inplace=True)

# Convert reservation_status_date to datetime
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%d-%m-%Y')

# Save cleaned data
df.to_csv('cleaned_hotel_bookings.csv', index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate total stay nights and revenue
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['revenue'] = df['adr'] * df['total_nights']

# Group by year and month for revenue trends
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-01')
revenue_trends = df[df['is_canceled'] == 0].groupby('arrival_date')['revenue'].sum()

# Plot
plt.figure(figsize=(10, 6))
revenue_trends.plot()
plt.title('Revenue Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.savefig('revenue_trends.png')
plt.show()


cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
print(f"Cancellation Rate: {cancellation_rate:.2f}%")

geo_distribution = df[df['is_canceled'] == 0]['country'].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=geo_distribution.index, y=geo_distribution.values)
plt.title('Top 10 Countries by Booking Count')
plt.xlabel('Country')
plt.ylabel('Number of Bookings')
plt.savefig('geo_distribution.png')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['lead_time'], bins=30, kde=True)
plt.title('Booking Lead Time Distribution')
plt.xlabel('Lead Time (days)')
plt.ylabel('Frequency')
plt.savefig('lead_time_distribution.png')
plt.show()


avg_revenue = df[df['is_canceled'] == 0]['revenue'].mean()
print(f"Average Revenue per Booking: ${avg_revenue:.2f}")

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load pre-trained embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert dataframe rows to text for embedding
texts = df.apply(lambda row: f"Hotel: {row['hotel']}, Canceled: {row['is_canceled']}, Revenue: {row['revenue']}, Country: {row['country']}, Date: {row['arrival_date']}", axis=1).tolist()
embeddings = embedder.encode(texts, show_progress_bar=True)

# Create FAISS index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

# Load LLM for generation
qa_model = pipeline('question-answering', model='mistralai/Mixtral-8x7B-v0.1', tokenizer='mistralai/Mixtral-8x7B-v0.1')

# Function to answer questions
def answer_question(query):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 matches
    context = "\n".join([texts[i] for i in I[0]])
    response = qa_model(question=query, context=context)
    return response['answer']

# Example usage
print(answer_question("Show me total revenue for July 2017"))


from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Load data
df = pd.read_csv('cleaned_hotel_bookings.csv')
df['arrival_date'] = pd.to_datetime(df['arrival_date'])

@app.post("/analytics")
def get_analytics():
    revenue_trends = df[df['is_canceled'] == 0].groupby('arrival_date')['revenue'].sum().to_dict()
    cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
    geo_dist = df[df['is_canceled'] == 0]['country'].value_counts().head(10).to_dict()
    return {
        "revenue_trends": revenue_trends,
        "cancellation_rate": cancellation_rate,
        "geo_distribution": geo_dist
    }

@app.post("/ask")
def ask_question(query: str):
    answer = answer_question(query)  # From RAG section
    return {"query": query, "answer": answer}

# Run with: uvicorn main:app --reload

import time

start = time.time()
answer = answer_question("What is the average price of a hotel booking?")
end = time.time()
print(f"Response Time: {end - start:.2f} seconds")