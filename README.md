# 🧠 Enhancing Developer Support: Automated Tagging on Stack Overflow

This project is a web-based application that predicts relevant Stack Overflow tags for a given programming-related question. It includes data scraping, cleaning, model training, and a real-time Streamlit app for inference.

---

## 📂 Project Structure

```text

├── web_scrapping.ipynb        # Web scraping notebook
├── app1.ipynb                 # Data cleaning and preprocessing notebook
├── best.ipynb                 # Model training and evaluation notebook
├── app.py                     # Streamlit web application
├── clean.csv                  # Cleaned dataset
└── README.md                  # Project documentation
```

## ⚙️ Setup Instructions (Windows with Conda)
### 1. Clone the repository

```
git clone https://github.com/VodnalaNaveen/Enhancing_Developer_Support_Automated_Tagging_on_Stack_Overflow.git
cd Enhancing_Developer_Support_Automated_Tagging_on_Stack_Overflow
```

### 2. Create and activate a Conda environment

```
conda create --name tag-predictor python=3.10 -y
conda activate tag-predictor

```

### 3. Install dependencies

```
pip install -r requirements.txt

```
Sample `requirements.txt`

```

streamlit
pandas
scikit-learn
beautifulsoup4
requests
```

## 🚀 Workflow
### 🕸️ 1. Web Scraping
  * Open `web_scrapping.ipynb`
  * Scrape Stack Overflow questions and tags using `requests` and `BeautifulSoup`
  * Store data for later processing

### 🧹 2. Data Cleaning & Preprocessing
  * Open `app1.ipynb`
  * Clean and preprocess the scraped data:
      * Handle missing values
      * Combine title and body
      * Format tags into list format
  * Save the cleaned data as `clean.csv`

### 🤖 3. Model Training

  * Open `best.ipynb`

  * Train a multi-label classification model using:

    * `TfidfVectorizer` for converting text into features

    * `OneVsRestClassifier` with `LogisticRegression`

  * Optionally save the trained pipeline

### 🌐 4. Launch Streamlit App

After cleaning and training, run the app using:

```
streamlit run app.py
```

The Streamlit interface will:

  * Accept a programming-related question as input
  * Predict relevant tags using the trained model
  * Display the predicted tags in real-time

## 🧠 Model Details
  * Text Vectorization: TF-IDF (1–2 grams), max features = 10,000
  * Classifier: OneVsRest with Logistic Regression

## 📌 Key Highlights
  * End-to-end machine learning pipeline: Scraping → Cleaning → Training → Inference
  * Built using Python, scikit-learn, Streamlit, BeautifulSoup, and requests
  * Real-time, interactive UI powered by Streamlit
  * Automatically suggests accurate Stack Overflow tags for programming questions
