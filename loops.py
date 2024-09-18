import streamlit as st

def app():
    st.title("Langchain with OpenAI & Ollama")

    st.subheader("Langchain with OpenAI")
    st.code("""
    # Import necessary libraries
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from the .env file
    load_dotenv()
    
    # Set environment variables for API keys (OpenAI and Langchain)
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['LANGCHAIN_TRACING_V2'] = "true"  # Enable Langchain tracing
    os.environ['LANGCHAIN_PROJECT'] = "GENAI APPLICATION WITH OPENAI"  # Define project name
    
    # Data Ingestion: Scraping data from the specified website
    from langchain_community.document_loaders import WebBaseLoader
    
    # Initialize the web scraper with the target URL
    loader = WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
    
    # Load documents from the website
    docs = loader.load()
    
    # Document Processing: Splitting documents into smaller chunks for efficient processing
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create a text splitter with a chunk size of 1000 characters and an overlap of 200 characters between chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Split the loaded documents into smaller chunks
    documents = text_splitter.split_documents(docs)
    
    # Embedding generation: Converting documents into vector representations
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    # Initialize the OpenAI embedding model to convert text into vector embeddings
    embeddings = OpenAIEmbeddings()
    
    # Vector Store: Storing the vectorized documents in a FAISS-based vector store for retrieval
    from langchain.vectorstores import FAISS
    
    # Create a FAISS vector store from the documents using the generated embeddings
    vectorstoredb = FAISS.from_documents(documents, embeddings)
    
    # Language Model Initialization: Setting up the OpenAI GPT-4 model for processing queries
    from langchain_openai import ChatOpenAI
    
    # Initialize the GPT-4 model as the LLM for query answering
    llm = ChatOpenAI(model="gpt-4o")
    
    # Retrieval Chain: Setting up a chain for retrieving and processing documents to answer queries
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.prompts import ChatPromptTemplate
    
    # Create a custom prompt template for document-based question answering
    prompt = ChatPromptTemplate.from_template(
        '''
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        '''
    )
    
    # Create a document processing chain that uses the LLM and custom prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retriever: Set up the retriever to retrieve relevant documents from the vector store
    retriever = vectorstoredb.as_retriever()
    
    # Create a retrieval chain that combines document retrieval and processing
    from langchain.chains import create_retrieval_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Invoke the retrieval chain to answer a specific question based on the retrieved documents
    response = retrieval_chain.invoke({"input": "LangSmith has two usage limits: total traces and extended"})
    
    # Output the generated answer from the LLM response
    print(response['answer'])
    """)

    st.subheader("Langchain with Ollama")
    st.code("""
    # Import necessary libraries
    import os
    from dotenv import load_dotenv
    import streamlit as st  # For building the web interface
    from langchain_community.llms import Ollama  # For using the Llama model
    from langchain_core.prompts import ChatPromptTemplate  # For creating prompt templates
    from langchain_core.output_parsers import StrOutputParser  # For parsing LLM output
    
    # Load environment variables from the .env file
    load_dotenv()
    
    # LangSmith Tracking setup (for tracing and logging)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # API key for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable tracing for LangSmith
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")  # Set project name
    
    # Create the prompt template for the LLM
    # The prompt has a system message for setting the AI's behavior and a user message for the question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the question asked."),
            ("user", "Question: {question}")
        ]
    )
    
    # Streamlit UI setup
    # Display the title of the app
    st.title("Langchain Demo With Llama 3.1 Model")
    
    # Create an input field where users can type their question
    input_text = st.text_input("What question do you have in mind?")
    
    # Initialize the Llama model (Ollama class) with the specified version
    llm = Ollama(model="llama3.1:latest")
    
    # Initialize the output parser to extract and format the LLM's response
    output_parser = StrOutputParser()
    
    # Create a chain that takes the prompt, processes it through the LLM, and then parses the output
    chain = prompt | llm | output_parser
    
    # If the user has entered a question, invoke the chain and display the result
    if input_text:
        # The 'invoke' function processes the input through the entire chain and outputs the answer
        st.write(chain.invoke({"question": input_text}))
    """)
