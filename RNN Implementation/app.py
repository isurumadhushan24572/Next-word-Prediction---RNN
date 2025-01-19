import streamlit as st
import numpy as np
import tensorflow as tf

# Load the pre-trained model and tokenizer
final_model = tf.keras.models.load_model("Next_word_Prediction.h5")
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Load and prepare the text data
path = "../WebScraping/paragraphs.txt"
with open(path, 'r', encoding='utf-8') as file:
    football_history = file.read()

tokenizer.fit_on_texts([football_history])

# Set the maximum sequence length
max_length = 337

# Streamlit UI
st.title("⚡ Next Word Predictor - RNN Model")
st.write("🎯 **Transform your ideas into complete sentences with the power of RNNs!**")
st.markdown(
    """
    This app predicts the next set of words based on the text you provide. 
    The model is trained on a dataset of football history paragraphs which is get through the web scraping.
    """
)

# Input for user text
st.subheader("🔤 Enter Your Starting Text")
input_text = st.text_input(
    "Start typing below and let Model take over!", 
    placeholder="Enter your text here..."
)

# Input for the number of words to predict
st.subheader("📈 Customize the Prediction")
no_of_words_predict = st.slider(
    "Select the number of words you want to predict:",
    min_value=1, 
    max_value=50, 
    value=6
)

# Button to trigger prediction
if st.button("🚀 Predict Next Words"):
    if input_text.strip():
        try:
            # Generate predictions
            result_text = input_text
            for i in range(no_of_words_predict):
                token_list = tokenizer.texts_to_sequences([result_text])[0]
                token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_length-1, padding='pre')
                predicted = np.argmax(final_model.predict(token_list), axis=-1)

                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break

                result_text += " " + output_word

            st.success("🌟 Your Predicted Text:")
            st.write(result_text)
        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter some initial text to start the prediction.")

# Footer
st.write("\n---\n")
st.markdown(
    """
    Developed  by **Isuru Madhushan**  
    
    """
)
