import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

os.makedirs("visuals",exist_ok=True)

df = pd.read_csv("7817_1.csv")

df = df[['reviews.text']]

df.dropna(inplace=True)


def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['reviews.text'].apply(get_sentiment)

print(df.head())

sentiment_counts = df['Sentiment'].value_counts()

print("\nSentiment Distribution")
print(sentiment_counts)

plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%'
)
plt.title("Sentiment Distribution")
plt.savefig("visuals/sentiment_pie.png")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(
    x='Sentiment',
    data=df
)
plt.title("Sentiment Count")
plt.tight_layout()
plt.savefig("visuals/sentiment_bar.png")
plt.show()

text = " ".join(df['reviews.text'].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.tight_layout()
plt.savefig("visuals/wordcloud.png")
plt.show()