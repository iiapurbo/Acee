import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from litellm import completion
from vector_store import get_vectorstore, get_retriever
import warnings

# Suppress huggingface_hub FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

st.set_page_config(page_title="বাংলা RAG এজেন্ট", layout="wide")

# Set Gemini API key
os.environ['GEMINI_API_KEY'] = "AIzaSyDehjBgSUtoDnQ7I4eMlwKBeVfkNMZ3rEs"

# Function to create RAG chain
def create_rag_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Define prompt template
        prompt_template = """
        আপনি একটি সহায়ক এজেন্ট। আপনার কাজ হলো শুধুমাত্র প্রদত্ত ডকুমেন্টের তথ্যের ভিত্তিতে প্রশ্নের উত্তর দেওয়া। 
        ডকুমেন্টের বাইরে কোনো তথ্য বা অনুমান ব্যবহার করবেন না। যদি প্রশ্নের উত্তর ডকুমেন্টে না থাকে, তাহলে বলুন যে তথ্য পাওয়া যায়নি। 
        উত্তর শুধুমাত্র বাংলায় দিন, প্রশ্ন যে ভাষায়ই হোক না কেন। সংখ্যা বা তথ্য ভুল বা কল্পনা করবেন না। 

        **প্রশ্ন**: {question}

        **ডকুমেন্ট থেকে প্রাসঙ্গিক তথ্য**: {context}

        **উত্তর**:
        """
        prompt = PromptTemplate.from_template(prompt_template)

        # Custom LLM function using LiteLLM
        def llm_function(inputs):
            question = inputs["question"]
            context = inputs["context"]
            formatted_prompt = prompt_template.format(question=question, context=context)
            response = completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            return response.choices[0].message.content

        # Create RAG chain using RunnableParallel
        rag_chain = RunnableParallel(
            {
                "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
                "question": RunnablePassthrough()
            }
        ) | RunnablePassthrough.assign(
            answer=lambda x: llm_function(x)
        ) | (lambda x: x["answer"])

        return rag_chain
    except Exception as e:
        raise Exception(f"RAG চেইন তৈরিতে ত্রুটি: {str(e)}")

# Streamlit interface
def main():
    
    st.title("বাংলা RAG ভিত্তিক প্রশ্নোত্তর সিস্টেম")
    st.write("এই সিস্টেম শুধুমাত্র নির্দিষ্ট .md ফাইলের তথ্যের ভিত্তিতে বাংলায় উত্তর দেয়।")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    # Initialize vectorstore and RAG chain
    if not st.session_state.initialized:
        try:
            #file_path = "./39.md"
            #if not os.path.exists(file_path):
             #   st.error(f"ফাইল {file_path} পাওয়া যায়নি। দয়া করে ফাইলটি সঠিক পথে রাখুন।")
              #  return
            
            # Load and process the .md file
            #splits = load_and_process_md(file_path)
            
            # Initialize vectorstore
            st.session_state.vectorstore = get_vectorstore()
            
            # Create RAG chain
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
            
            st.session_state.initialized = True
            st.success("ডকুমেন্ট লোড, এমবেডিং এবং ইনডেক্সিং সফলভাবে সম্পন্ন হয়েছে!")
        except Exception as e:
            st.error(f"ডকুমেন্ট লোড বা ইনিশিয়ালাইজেশনে সমস্যা: {str(e)}")
            return

    # Chat interface
    st.subheader("আপনার প্রশ্ন জিজ্ঞাসা করুন")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input via text area
    user_query = st.text_area("আপনার প্রশ্ন লিখুন (ইংরেজি বা বাংলায়):", height=100, key="query")
    
    if st.button("প্রশ্ন জমা দিন"):
        if not user_query:
            st.warning("দয়া করে একটি প্রশ্ন লিখুন।")
            return
        
        if 'rag_chain' not in st.session_state:
            st.error("RAG চেইন ইনিশিয়ালাইজ করা হয়নি। দয়া করে ফাইলটি সঠিক পথে রাখুন এবং স্ক্রিপ্টটি পুনরায় চালান।")
            return
        
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        try:
            with st.spinner("উত্তর তৈরি করা হচ্ছে..."):
                response = st.session_state.rag_chain.invoke(user_query)
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        except Exception as e:
            st.error(f"উত্তর তৈরিতে সমস্যা: {str(e)}")

if __name__ == "__main__":
    main()