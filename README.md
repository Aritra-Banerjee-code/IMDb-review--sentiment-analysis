# IMDb-review--sentiment-analysis
This project performs sentiment analysis on the IMDb movie reviews dataset (LMST) to classify reviews as positive or negative.
It leverages Natural Language Processing (NLP) techniques and machine learning/deep learning models to understand the sentiment expressed in text reviews.

#🚀 Features

Preprocessing of IMDb reviews (tokenization, stopword removal, padding, etc.)

Model building using:

Traditional ML models (Logistic Regression, Naive Bayes, SVM)

Deep Learning (LSTM/GRU, Bi-LSTM) for sequential text data

Evaluation with accuracy, precision, recall, and F1-score

Visualization of training curves and word distributions

Support for custom review predictions

#📂 Dataset

The project uses the IMDb Large Movie Review Dataset (LMST) containing:

50,000 labeled reviews (25k positive, 25k negative)

Balanced dataset for binary sentiment classification

#🛠️ Tech Stack

Python (NumPy, Pandas, Matplotlib, Seaborn)

NLP libraries: NLTK / spaCy

Deep Learning: TensorFlow / PyTorch

Machine Learning: Scikit-learn

#📊 Results

Achieved up to ~90% accuracy using an LSTM-based deep learning model.

Traditional ML models achieved ~80–85% accuracy depending on feature extraction (Bag of Words / TF-IDF).
