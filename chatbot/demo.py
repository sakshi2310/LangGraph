import streamlit as st

with st.chat_message("user"):
    st.text("Hi")

with st.chat_message("ai"):
    st.text("hi sakshi")

with st.chat_message("human"):
    st.text("how are you")

user_inpu = st.chat_input("Type here :")

if user_inpu:
    with st.chat_message('user'):
        st.text(user_inpu)