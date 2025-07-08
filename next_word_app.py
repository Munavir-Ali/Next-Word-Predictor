import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.pikbest.com/wp/202405/nodes-1-3d-render-of-connected-by-lines-on-a-black-background_9857780.jpg!sw800");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#st.markdown("<h1 style='color:blue;'>Next Word Predictor (LSTM)</h1>", unsafe_allow_html=True)
#st.markdown("<p style='color:green;'>Enter a sequence of words, and the model will predict the next word.</p>", unsafe_allow_html=True)

st.title("Next Word Predictor (LSTM)")
st.write("Enter a sequence of words, and the model will predict the next word. Input a full stop to stop and display the sentence.")

#Load tokenizer and model
with open('next_word_prediction_tokens0.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('next_word_prediction_model_0.h5')
max_sequence_length = 55
top_k = 5


#Initialize session state
if 'sequence' not in st.session_state:
    st.session_state.sequence = []

if 'done' not in st.session_state:
    st.session_state.done = False

if 'last_input' not in st.session_state:
    st.session_state.last_input = ""


def get_next_word_suggestions(current_sequence):
    token_list = tokenizer.texts_to_sequences([current_sequence])[0]
    if not token_list:
        return []

    token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_indices = np.argsort(predicted_probs)[-top_k:][::-1]

    index_word = tokenizer.index_word
    predicted_words = [index_word.get(i, "?") for i in predicted_indices]

    return predicted_words

#Interactive Loop
if not st.session_state.done:
    current_text = " ".join(st.session_state.sequence)
    input_text = st.text_input("Type your sentence here", value=current_text)

    # Detect new word without requiring enter after each word
    if input_text != st.session_state.last_input:
        words = input_text.strip().split()

        #fullstop or new word entry
        if len(words) > len(st.session_state.sequence):
            new_word = words[-1]

            if new_word == ".":
                st.session_state.done = True
            else:
                st.session_state.sequence.append(new_word)

            st.session_state.last_input = input_text
            st.rerun()

        # Handle case when user deletes or rewrites sentence
        elif len(words) < len(st.session_state.sequence):
            st.session_state.sequence = words
            st.session_state.last_input = input_text
            st.rerun()

        else:
            st.session_state.last_input = input_text

    #clickable suggestions
    suggestions = get_next_word_suggestions(input_text)
    if suggestions:
        st.write("### Suggestions:")
        cols = st.columns(top_k)
        for i in range(top_k):
            if cols[i].button(suggestions[i]):
                st.session_state.sequence.append(suggestions[i])
                st.session_state.last_input = " ".join(st.session_state.sequence)
                st.rerun()

#Final Display of Entered Sentence
else:
    full_sentence = " ".join(st.session_state.sequence) + "."
    st.success("Final Sentence:")
    st.write(f"`{full_sentence}`")

    if st.button("Reset"):
        st.session_state.sequence = []
        st.session_state.done = False
        st.session_state.last_input = ""
        st.rerun()
