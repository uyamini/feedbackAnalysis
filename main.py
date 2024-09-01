from src.data_extraction import fetch_feedback_data
from src.text_preprocessing import preprocess_text
from src.sentiment_analysis import analyze_sentiments_roberta
from src.key_takeaways import extract_key_takeaways
from collections import Counter

def main():
    #Fetch feedback data from the database
    feedback_df = fetch_feedback_data()
    #Preprocess each feedback
    feedback_df['cleaned_text'] = feedback_df['feedback_text'].apply(preprocess_text)
    #Analyze sentiments for each feedback
    sentiment_labels = analyze_sentiments_roberta(feedback_df['cleaned_text'])
    #Add sentiment labels to the dataframe
    feedback_df['sentiment'] = sentiment_labels
    #Count the most common sentiments
    sentiment_counts = Counter(sentiment_labels)
    #Print each feedback entry with its sentiment
    for index, row in feedback_df.iterrows():
        print(f"Original Text: {row['feedback_text']}")
        print(f"Cleaned Text: {row['cleaned_text']}")
        print(f"Detected Sentiment: {row['sentiment']}\n")
    #Print the most common sentiments
    print("Most Common Sentiments:")
    for sentiment, count in sentiment_counts.most_common():
        print(f"{sentiment.capitalize()}: {count}")

    #Combine feedback to extract key takeaways
    combined_feedback = ' '.join(feedback_df['cleaned_text'])
    key_takeaways = extract_key_takeaways([combined_feedback])

    #Display key takeaways
    print("\nKey Takeaways:")
    for takeaway in key_takeaways:
        print(f"- {takeaway}")

if __name__ == "__main__":
    main()
