import streamlit as st
from pypdf import PdfReader 
from chatbot import ChatBot

assistant = ChatBot()


def read_pdf(file):
    text=''
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text


st.set_page_config(page_title='Simple Chatbot')
st.title('ChatBot')
placeholder = st.empty()

if 'messages' not in st.session_state:
    st.session_state.messages=[]

with st.sidebar:
    st.markdown(' Select pdf book ')
    upload = st.file_uploader('',type=['pdf'])
    if upload:
        btn = st.button('Process')
        if btn:
            with st.spinner('Processing'):
                assistant.update(read_pdf(upload))
                placeholder.success('Ready to Chat')
                # flush session state
                st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Enter your question')
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    try:
        answer = assistant.ask(prompt)
        with st.chat_message('assistant'):
            st.markdown(answer)
    except:
        placeholder.error('No pdf Uploaded ')
    

    st.session_state.messages.append({'role':'assistant','content':answer})
