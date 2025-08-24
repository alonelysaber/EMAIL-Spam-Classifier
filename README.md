# Email Spam Classification using Random Forest

A machine learning project that classifies emails as spam or ham (legitimate) using Natural Language Processing techniques and Random Forest classifier.

## Overview

This project implements a spam email detection system that:
- Preprocesses email text data using NLP techniques
- Uses bag-of-words representation for feature extraction
- Employs Random Forest classifier for prediction
- Achieves ~97.5% accuracy on test data

## Features

- **Text Preprocessing**: Removes punctuation, converts to lowercase, removes stopwords, and applies stemming
- **Feature Engineering**: Uses CountVectorizer for bag-of-words representation
- **Machine Learning**: Random Forest classifier with parallel processing
- **High Accuracy**: Achieves 97.49% accuracy on test dataset

## Requirements

```python
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

## Installation

1. Clone the repository or download the notebook
2. Install required packages:
```bash
pip install pandas numpy scikit-learn nltk
```
3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Dataset

The project expects a CSV file named `spam_ham_dataset.csv` with the following structure:
- `text`: Email content
- `label_num`: Numerical labels (0 for ham, 1 for spam)

## Usage

### 1. Data Loading and Preprocessing
```python
# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')

# Clean text data
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))
```

### 2. Text Processing Pipeline
The preprocessing pipeline includes:
- Converting text to lowercase
- Removing punctuation
- Removing English stopwords
- Applying Porter stemming
- Creating a clean corpus

### 3. Feature Extraction
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num
```

### 4. Model Training
```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest classifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
```

### 5. Model Evaluation
```python
# Check accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### 6. Making Predictions
```python
# Preprocess new email
def preprocess_email(email_text):
    email_text = email_text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    return ' '.join(email_text)

# Transform and predict
email_corpus = [preprocess_email(new_email)]
X_email = vectorizer.transform(email_corpus)
prediction = clf.predict(X_email)
```

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.49%
- **Features**: Bag-of-words representation with stemming and stopword removal
- **Cross-validation**: 80-20 train-test split

## File Structure

```
project/
│
├── main.ipynb                 # Main Jupyter notebook
├── spam_ham_dataset.csv       # Dataset file
└── README.md                  # This file
```

## Key Components

### Text Preprocessing
1. **Normalization**: Convert to lowercase and remove line breaks
2. **Punctuation Removal**: Strip all punctuation marks
3. **Stopword Filtering**: Remove common English words
4. **Stemming**: Reduce words to root forms using Porter Stemmer

### Feature Engineering
- **Bag-of-Words**: CountVectorizer creates numerical representations
- **Sparse Matrix**: Efficient storage of high-dimensional feature vectors

### Classification
- **Random Forest**: Ensemble method with parallel processing (`n_jobs=-1`)
- **Robust Performance**: Handles high-dimensional sparse data well

## Potential Improvements

1. **Advanced NLP**: Use TF-IDF instead of simple count vectorization
2. **Deep Learning**: Implement LSTM or BERT-based models
3. **Feature Engineering**: Add email metadata features (sender, subject length, etc.)
4. **Hyperparameter Tuning**: Optimize Random Forest parameters
5. **Cross-Validation**: Implement k-fold cross-validation for better evaluation

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Run `nltk.download('stopwords')` if you get download errors
2. **Memory Issues**: For large datasets, consider using sparse matrices throughout
3. **Encoding Problems**: Ensure your CSV file uses UTF-8 encoding

### Dependencies
Make sure all required packages are installed with compatible versions:
- pandas >= 1.0.0
- scikit-learn >= 0.24.0
- nltk >= 3.5

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This is a educational project demonstrating basic NLP and machine learning techniques for spam classification. For production use, consider more advanced preprocessing, feature engineering, and model validation techniques.
