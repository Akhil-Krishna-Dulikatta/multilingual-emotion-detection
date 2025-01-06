# multilingual-emotion-detection
Multilingual Emotion Detection Using NLP
Welcome to the Multilingual Emotion Detection repository! This project focuses on leveraging Natural Language Processing (NLP) to detect emotions expressed in different languages with high precision and recall. The repository implements two models: BERT (a state-of-the-art deep learning model) and SVM (a traditional machine learning model) for multilingual emotion classification.

Project Overview
Motivation
In a globalized world, understanding emotions across languages is pivotal for fostering effective communication and empathy. This project aims to create a robust NLP model that can detect emotions in multilingual text data, overcoming linguistic barriers and cultural nuances.

Objectives
Develop a model that accurately captures emotions in multilingual text.
Ensure consistent emotion classification across diverse languages and contexts.
Compare the performance of deep learning (BERT) and traditional machine learning (SVM) models for emotion detection.
Features
Multilingual Emotion Detection: Supports various languages, including English, French, German, Spanish, Italian, Russian, and more.
High Accuracy: BERT model achieves near-perfect accuracy (99.85%) in detecting emotions.
Cross-Model Comparison: Includes a detailed performance comparison between BERT and SVM models.
Custom Visualizations: Confusion matrices and aggregated emotion distribution charts.
Dataset
The dataset used includes multilingual text samples labeled with emotions such as Joy, Anger, Sadness, Fear, and Surprise. The dataset has been preprocessed to map all emotions to English equivalents for consistency.

Emotion Mapping Example:
Original Emotion	English Equivalent
Hněv (Czech)	Anger
Gioia (Italian)	Joy
Грусть (Russian)	Sadness
Implementation Workflow
Dataset Collection: Collect multilingual text labeled with emotions.
Data Preprocessing: Clean, tokenize, and normalize the text data.
Feature Engineering: Convert text into numerical features using TF-IDF and BERT tokenization.
Model Training:
BERT Model: Fine-tuned for sequence classification.
SVM Model: Trained on TF-IDF features.
Evaluation & Visualization:
Confusion matrices to evaluate model predictions.
Aggregated emotion distributions for insights.
Comparison: Analyze and compare the performance of BERT and SVM models.
Models
1. BERT
Pre-trained bert-base-multilingual-cased model fine-tuned for emotion detection.
High precision and recall across all emotion categories.
Best suited for tasks requiring context understanding.
2. SVM
Trained using TF-IDF features.
Provides a robust baseline for comparison.
Effective for simpler tasks with structured feature spaces.
Performance Metrics
Accuracy: BERT reached 99.85% on validation data.
Precision, Recall, and F1-Score: Achieved near-perfect scores across all emotions for both models.
Visualization: Confusion matrices and emotion distribution charts.
Setup and Usage
Prerequisites
Python 3.8+
Required libraries: pandas, numpy, tensorflow, transformers, scikit-learn, matplotlib, seaborn
Installation

Clone the repository:
bash

git clone https://github.com/AkhilKrishnaDulikatta/multilingual-emotion-detection.git
cd multilingual-emotion-detection
Install dependencies:
bash

pip install -r requirements.txt
Running the Code
Upload your dataset in .csv format.
Execute the Python script in Google Colab or your local environment:
bash

python emotion_detection.py
View results and visualizations for both BERT and SVM models.
Visualizations
BERT Confusion Matrix
Shows the model's prediction accuracy for each emotion.

Aggregated Emotion Distribution
Bar chart displaying the frequency of detected emotions.

Results
BERT Model: Exceptional accuracy and context comprehension.
SVM Model: Reliable baseline with robust performance.
Key Insights
Emotion Diversity: The dataset captures a wide range of emotions across languages.
Cultural Context: Highlights variations in emotional expression between languages.
Model Adaptability: BERT performs exceptionally well in multilingual settings.
Limitations
Overfitting Risk: Near-perfect accuracy could indicate overfitting.
Low-Resource Languages: Limited data for certain languages affects generalization.
Visualization Challenges: Glyph rendering issues with multilingual character sets.
Future Directions
Expand the dataset to include more low-resource languages.
Enhance tokenization and encoding techniques.
Investigate additional deep learning architectures for emotion detection.
Contributing
Contributions are welcome! Please fork the repository and create a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For any questions or feedback, please contact:

Akhil Krishna Dulikatta
Email - akhilkrishnadulikatta1@gmail.com | LinkedIn - https://www.linkedin.com/in/akhil-krishna-dulikatta/

Happy coding! 🚀
