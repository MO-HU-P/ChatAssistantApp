import streamlit as st
from streamlit_chat import message
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv

load_dotenv()

class ChatHistory:

# Initialization method for the class managing chat history. Initializes the history if it doesn't exist in the session state.
    def __init__(self):
        if "history" not in st.session_state:
            st.session_state["history"] = []

# Method for displaying chat messages. Displays the chat history in reverse order, alternating between user and assistant messages.
    def display_messages(self):
        for i in reversed(range(len(st.session_state["history"]))):
            message = st.session_state["history"][i]
            if i % 2 == 0:
               with st.chat_message("user"):
                    st.markdown(message)
            else:
               with st.chat_message("assistant"):
                    st.markdown(message)

# Method to display the number of tokens in chat messages. Displays the total token count in the sidebar.
    def display_tokens(self):
        self.all_message = []   
        for message in st.session_state["history"]:

            tiktoken_encoding = tiktoken.encoding_for_model(self.model_selection)
            encoded = tiktoken_encoding.encode(message)
            tokens_in_message = len(encoded)
            
            self.all_message.append(tokens_in_message)
        
        total_tokens = sum(self.all_message)
        st.sidebar.info(f"Total tokens: {total_tokens}")


class ChatAssistantApp: 
    page_A = "Text Generation Page"
    page_B = "PDF Chat Page"
    
# Initialization method for the Chat Assistant application. Initializes the history if it doesn't exist in the session state.    
    def __init__(self):
        ChatHistory.__init__(self)
        self.tokens = 0

# Method to display the application title.
    def show_title(self):
        st.title("🤗 Chat Assistant App")

# Method to display API key authentication in the sidebar. Handles error if the API key is not provided.  
    def show_sidebar(self):
        st.sidebar.header("API Key Authentication")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
           self.error_handling()
        else:
            st.sidebar.success("API key submitted.")

# Method to handle API key errors. Displays a warning and error message if the API key is not provided. Provides an option to paste the API key directly into a text box.
    def error_handling(self):
        if st.sidebar.checkbox('You can paste the API key directly into the text box.'):
           os.environ['OPENAI_API_KEY']  = st.sidebar.text_input(label="OpenAI API key", type="password")
           openai.api_key = os.environ['OPENAI_API_KEY']
           if openai.api_key: 
              st.sidebar.success("API key submitted.")
        else:
           st.sidebar.warning("API key not provided.")
           st.sidebar.error("Unable to execute. Please check your OpenAI API key.")              

# Method to select a page in the application. Page A focuses on text generation, while page B allows PDF information to be included in the chat.
    def page_select(self):
        self.selected_page = st.sidebar.selectbox("Select a page", [self.page_A, self.page_B])
        if self.selected_page == self.page_A:
            st.write(f"Welcome to the {self.page_A}! The main focus here is on text generation.")
        else:
            st.write(f"Welcome to the {self.page_B}! This page allows you to include PDF information in your chat.")

# Method to configure the page based on the selected page. Sets model and parameter selections.
    def page_config(self):
        col1, col2 = st.columns([2, 1])
        
        if self.selected_page == self.page_A:
            with col1:
                self.model_selection = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"], index=0)
                self.chat_behavior = st.text_input("Chat Behavior:", "You are a helpful assistant.", key="chat_behavior", help="System message for Chatbot")
            with col2:
                self.temperature_degree = st.slider("Temperature", 0.0, 2.0, 0.5, step=0.1)
                self.max_tokens = st.number_input("Max Tokens", min_value=1, value=500, step=1)

        if self.selected_page == self.page_B:
            with col1:
                self.model_selection = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"], index=0)
            with col2:        
                self.temperature_degree = st.slider("Temperature", 0.0, 2.0, 0.5, step=0.1) 
                
            self.upload_pdf()

# Method to handle the upload of a PDF file. Continues processing if a PDF file is detected.
    def upload_pdf(self):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

        if uploaded_file is None:
            st.warning("PDF file is not recognized.")
        else:
            st.success("PDF file is detected.")
            self.process_pdf(uploaded_file)

# Method to process the uploaded PDF file and initialize the ConversationalRetrievalChain.
    def process_pdf(self, uploaded_file):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 100,
        length_function = len,
        )       
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name 
        try:
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load_and_split(text_splitter)
            embeddings = OpenAIEmbeddings()
            vectors = FAISS.from_documents(data, embeddings)

            memory = self.save_memory()
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=self.temperature_degree, model_name=self.model_selection),
                retriever=vectors.as_retriever(), memory=memory
                )
        finally:
            os.remove(tmp_file_path)
            st.write("Please feel free to ask any questions.")

# Method to initialize the memory for saving the chat message history.
    def save_memory(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory

# Method to initialize the chat message history.
    def initialize_history(self):
        self.tokens = 0
        clear_all_button = st.button("Clear Conversation")
        if clear_all_button:
           st.session_state["history"] = []

# Method to process user input. Performs text generation or conversational chat based on user input.
    def process_user_input(self):
        container = st.container()
        with container:
            with st.form(key='text_form', clear_on_submit=True):
                 self.user_input = st.text_area(label='Your Message:', key='input', height=100)
                 submit_button = st.form_submit_button(label='Send')
           
            if submit_button and self.user_input:
               if self.selected_page == self.page_A: 
                  st.session_state["history"].append(self.user_input) 
                  self.text_generation()
               else:
                  st.session_state["history"].append(self.user_input) 
                  self.conversational_chat() 

# Method to process conversational chat. Uses cache for repeated questions and responds to new questions.
    def conversational_chat(self):
        existing_question = next((entry for entry in st.session_state["history"] if entry[0] == self.user_input), None)
        if not existing_question:
           result = self.chain({"question": self.user_input, "chat_history": st.session_state["history"]})

           st.session_state["history"].append(result["answer"])

           ChatHistory.display_messages(self)
           ChatHistory.display_tokens(self)

           return result["answer"]

# Method to process text generation. Generates text based on user input.
    def text_generation(self):
            try:
              messages_for_api = [{"role": "system", "content": self.chat_behavior}]
              messages_for_api += [{"role": "user", "content": self.user_input} for i in range(0, len(st.session_state["history"]))]

              response = openai.ChatCompletion.create(
                  model=self.model_selection,
                  messages=messages_for_api,
                  temperature=self.temperature_degree,
                  max_tokens=self.max_tokens
              )

              st.session_state["history"].append(response['choices'][0]['message']['content'])
              
              ChatHistory.display_messages(self)
              ChatHistory.display_tokens(self)

            except Exception as e:
              st.error(f"Error executing OpenAI API request: {str(e)}")
              st.stop()

# Method to execute the application. Displays the title, sidebar, page selection, page configuration, initializes history, and processes user input.
    def run_app(self):
        self.show_title()
        self.show_sidebar()
        self.page_select()
        self.page_config()
        self.initialize_history()
        self.process_user_input()

if __name__ == "__main__":
    app = ChatAssistantApp()
    app.run_app()



