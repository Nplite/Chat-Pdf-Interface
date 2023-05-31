import streamlit as st
from dotenv import load_dotenv
import pickle
import altair as alt
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from ttsmms import TTS
import IPython.display as ipd


with st.sidebar:
    st.title('🤗💬 Varahi Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')

load_dotenv()


def main():
    os.environ["OPENAI_API_KEY"] = "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"
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                
                response = chain.run(input_documents=docs, question=query)

            # Format predictions as numbered list
            predictions = response.split(". ")
            formatted_predictions = "\n".join([f"{i+1}. {prediction}" for i, prediction in enumerate(predictions)])

            st.write(formatted_predictions)


            # tts = TTS("data/eng")

            # if st.button("Synthesize and Play"):

            #     wav_text = tts.synthesis(formatted_predictions)

            #     ipd.Audio(wav_text["x"], rate=wav_text["sampling_rate"], autoplay=True)




if __name__ == '__main__':
    main()


