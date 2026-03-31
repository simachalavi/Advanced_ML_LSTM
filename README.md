# 🎯 Suicide Detection with Bi-LSTM and Attention

## 📖 Introduction

This project explores the detection of suicidal thoughts in text using deep learning. By leveraging a **Bidirectional LSTM** combined with a custom **Attention Mechanism**, the model identifies linguistic patterns of distress and hopelessness from social media posts.

The goal is to provide a high-accuracy classification tool that can distinguish between "Suicide" and "Non-Suicide" labels, serving as a foundational step for digital mental health monitoring.

## 🛠️ Project Architecture

The model uses a sophisticated NLP pipeline to process and classify text:

1.  **Text Preprocessing:** Tokenization, lowercase conversion, and punctuation removal.

2.  **Embedding Layer:** Maps words into a 200-dimensional dense vector space.

3.  **Bidirectional LSTM:** Processes sequences in both forward and backward directions to capture full context.

4.  **Attention Layer:** Assigns relative importance to specific words (tokens) within a post, "highlighting" key indicators of risk.

5.  **Classification Head:** A Dense layer with Dropout (0.315) for regularization, followed by a Sigmoid output.

6.  **Evaluation:** Using EarlyStopping and ModelCheckpoint for optimized training.

# 📊 Dataset

The data is sourced from the [Suicide Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) on Kaggle. It contains:

-   **Total Records:** \~232,074 rows.

-   **Source:** Scraped posts from the "SuicideWatch" and "depression" subreddits.

-   **Features:** Raw text posts and a class label (`suicide` or `non-suicide`).

# 🚀 Getting Started

## 1. Prerequisites (Docker)

This project is optimized for running within a GPU-enabled Docker container.

. **Image:** `tensorflow/tensorflow:2.14.0-gpu-jupyter2`.

## 2. Installation

To set up the environment locally or within your container, install the required dependencies:

``` bash
pip install -r requirements.txt
```

## 3. Running the Notebook

Open the notebook in your Jupyter environment:

``` bash
jupyter notebook notebooks/bi-lstm-with-attention_Final.ipynb
```

# 📈 Model Performance

-   **Loss Function:** Binary Crossentropy

-   **Optimizer:** Adam (Learning Rate: 1e-5)

-   **Key Metric:** Accuracy

|           |                  |              |
|-----------|------------------|--------------|
| **Layer** | **Output Shape** | **Param \#** |
| Input     | (None, 40)       | 0            |
| Embedding | (None, 40, 200)  | 8,000,000    |
| Bi-LSTM   | (None, 40, 60)   | 27,840       |
| Attention | (None, 60)       | 100          |
| Dense     | (None, 1)        | 61           |

# 🗂️ Repository Structure

``` plaintext
├── data\
├── notebooks/\
│ └── bi-lstm-with-attention_SC.ipynb \# Main Project Notebook 
├── requirements.txt \# Library dependencies 
└── README.md \# Project documentation 
```

# 🛡️ License & Ethics

This project is for educational purposes only. Automated suicide detection is a sensitive clinical task and should not be used as a standalone diagnostic tool without professional medical oversight.

# 🤝 Contributions

Feel free to open an issue or submit a pull request if you have ideas for improving the Attention mechanism or testing different word embeddings!