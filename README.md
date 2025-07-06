# 🧠 Next-Word Predictor

This project demonstrates a **Next Word Prediction** system built using **Deep Learning (LSTM)** and trained on the **Shakespeare dataset** (`alllines.txt`). It predicts the most likely next word in a sentence using AI/ML techniques. The frontend is built using **Streamlit** for easy interaction.

---

## 🚀 Key Features

### 🔹 Model Architecture
- Built with a **single-layer LSTM (Long Short-Term Memory)** network
- Designed to capture the temporal sequence of words in English text

### 🔹 Loss Function
- Utilizes **Sparse Categorical Crossentropy**
- Avoids one-hot encoding for improved efficiency and reduced memory usage

### 🔹 Dataset
- Trained on **Shakespeare's complete works** using `alllines.txt`
- A rich corpus of poetic and dramatic English text

### 🔹 Frontend
- Clean, interactive UI built with **Streamlit**
- Users can enter a sentence and receive **top-5 predictions** for the next word

---

## 🔧 Tools & Technologies Used
- Python 3
- TensorFlow / Keras (for model development)
- Streamlit (for frontend)
- NumPy & Pickle (for tokenization and data handling)
- Shakespeare Dataset (`alllines.txt`, from Kaggle)

---

## 📚 How It Works

- The input sentence is **tokenized** using a pre-trained tokenizer.
- The token sequence is **padded** to a fixed length (`max_sequence_length = 55`) before being passed into the LSTM model.
- The model uses a **single-layer LSTM** to predict the **top 5 most likely next words**, based on patterns learned from Shakespeare’s language.
- The predicted words are displayed as **clickable suggestions** on the Streamlit interface, allowing users to interactively build sentences.

---

## 💡 Use Cases
- Language modeling and NLP experimentation
- Learning and visualizing LSTM-based sequence modeling
- Educational demo for text prediction in real-time

---

> 👨‍💻 Feel free to clone, modify, and experiment with this project!
