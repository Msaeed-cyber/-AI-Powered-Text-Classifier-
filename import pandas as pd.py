import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import tkinter as tk
from tkinter import messagebox

# Prepare the data
positive_texts = [
    "This is a great movie!",
    "I love this product, it works perfectly.",
    "What a wonderful day!",
    "The customer service was excellent.",
    "I'm so happy with the results.",
    "This is an amazing experience.",
    "I feel fantastic today.",
    "The food was delicious.",
    "I'm so grateful for this opportunity.",
    "This is the best day ever!",
    "I love this movie!",
    "What a fantastic experience.",
    "Absolutely wonderful performance.",
    "I am very happy with the results.",
    "This is great news.",
    "I'm excited for the event.",
    "Everything is perfect.",
    "It's a beautiful day.",
    "I'm feeling awesome today.",
    "I really like this idea.",
    "This place is amazing!",
    "So proud of our team!",
    "This opportunity is excellent.",
    "Fantastic work by the developers.",
    "This is the most incredible thing I've ever seen!",
    "I am overjoyed with the outcome.",
    "Simply breathtaking!",
    "The result exceeded all my expectations.",
    "This is a truly inspiring moment.",
    "Feeling incredibly optimistic about the future.",
    "What a brilliant idea!",
    "I'm having the time of my life.",
    "This is pure bliss.",
    "So thankful for all the support.",
    "This made my day!"
]

negative_texts = [
    "I hate this, it's terrible.",
    "The service was very poor.",
    "This is the worst experience ever.",
    "I'm so disappointed.",
    "The quality is very low.",
    "I am not happy with your reply.",
    "This is unacceptable.",
    "I'm so frustrated.",
    "The product is defective.",
    "I regret buying this.",
    "I hate everything about this.",
    "Very bad service.",
    "It was a horrible experience.",
    "This is terrible.",
    "What a waste of time.",
    "Extremely poor quality.",
    "This made me really angry.",
    "Totally unacceptable behavior.",
    "Nothing worked as expected.",
    "The product broke after one use.",
    "Customer support was useless.",
    "I won’t recommend this to anyone.",
    "This is a complete disaster.",
    "I'm utterly disgusted.",
    "Absolutely dreadful.",
    "I feel completely let down.",
    "This is a scam.",
    "I'm so annoyed by this.",
    "This is a huge inconvenience.",
    "I can't stand this.",
    "This is so unfair.",
    "I'm really upset about it.",
    "This is a major failure."
]

neutral_texts = [
    "The weather is cloudy today.",
    "The meeting is scheduled for 2 PM.",
    "I need to buy some groceries.",
    "The report is on my desk.",
    "The car is blue.",
    "The sky is clear.",
    "I am going for a walk.",
    "The book is on the table.",
    "I need to check my email.",
    "The train arrives at noon.",
    "It's okay, nothing special.",
    "I don't have an opinion.",
    "Meh, it’s fine.",
    "I neither like it nor dislike it.",
    "It’s just another day.",
    "I’m not sure how I feel.",
    "The event was average.",
    "That’s neutral to me.",
    "I guess it was okay.",
    "Nothing exciting happened.",
    "This is standard procedure.",
    "Service was acceptable.",
    "I had mixed feelings.",
    "I felt indifferent.",
    "Not too good, not too bad.",
    "The experience was moderate.",
    "It met my expectations.",
    "No strong opinion here.",
    "I suppose it’s fine.",
    "It was somewhat expected.",
    "The data is being processed.",
    "The system is running.",
    "The document is attached.",
    "The information is available.",
    "The task is completed.",
    "The status is pending.",
    "The report will be generated.",
    "The file is saved.",
    "The process is ongoing.",
    "The results are in."
]
texts = positive_texts + negative_texts + neutral_texts
labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts) + ['neutral'] * len(neutral_texts)

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
model = LinearSVC()
model.fit(X, y)

# --- GUI Application ---
def classify_text():
    user_input = input_field.get()
    if not user_input.strip():
        messagebox.showwarning("Input Required", "Please enter some text.")
        return
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)
    result_label.config(text=f"Sentiment: {prediction[0].capitalize()}")

# Set up tkinter window
window = tk.Tk()
window.title("AI Text Sentiment Classifier")
window.geometry("400x250")
window.config(bg="#f0f0f0")

# UI Elements
title_label = tk.Label(window, text="Enter Text for Sentiment Classification", font=("Arial", 14), bg="#f0f0f0")
title_label.pack(pady=10)

input_field = tk.Entry(window, width=50, font=("Arial", 12))
input_field.pack(pady=10)

classify_button = tk.Button(window, text="Classify Sentiment", command=classify_text, font=("Arial", 12), bg="#4CAF50", fg="white")
classify_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

# Run app
window.mainloop()
