import streamlit as st
import numpy as np
from PIL import Image
from hugchat import hugchat
from hugchat.login import Login
from utility import edit_distance, process_image, annotate_image, MODEL, PROTOTXT


def word_correction():
    with open(r'.\source\data\vocab.txt', 'r') as f:
        vocabs = f.readlines()
        st.title('Word Correction using Levenshtein Distance')
        word = st.text_input('Word')
        if st.button("Compute"):
            edit_dis = dict()
            for vocab in vocabs:
                edit_dis[vocab] = edit_distance(word, vocab)
            sorted_dis = dict(sorted(edit_dis.items(), key=lambda x: x[1]))
            corrected_word = list(sorted_dis.keys())[0]
            st.write('Correct word: ', corrected_word)
            col1, col2 = st . columns(2)
            col1.write('Vocabulary :')
            col1.write(vocabs)
            col2.write('Distances :')
            col2.write(sorted_dis)


def object_detection():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")


def chat_bot():
    # App title
    st.title('Simple ChatBot')

    # Hugging Face Credentials
    with st.sidebar:
        st.title('Login HugChat')
        hf_email = st.text_input('Enter E-mail:')
        hf_pass = st.text_input('Enter Password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your account!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Function for generating LLM response
    def generate_response(prompt_input, email, passwd):
        # Hugging Face Login
        sign = Login(email, passwd)
        cookies = sign.login()
        # Create ChatBot
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    # User-provided prompt
    if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, hf_email, hf_pass)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


tab1, tab2, tab3 = st.tabs(['Word Correction', 'Object Detection', 'Chat Bot'])
with tab1:
    word_correction()
with tab2:
    object_detection()
with tab3:
    chat_bot()
