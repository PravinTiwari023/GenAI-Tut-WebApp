import streamlit as st


def app():
    """
    1. Adding Title and Subheader
    2. Adding Code
    3. Adding Sections with Required tabs section
        a. Data Ingestion Techniques
        b. Text Splitting Techniques
    """

    # Adding a title to the app
    st.title("Data Ingestion Techniques")

    # Creating tabs for different types of data loaders
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Text Loader", "PDF Loader", "Web Loader", "ArXiv Loader", "Wikipedia Loader"])

    # Text Loader Tab
    with tab1:
        """
        1. Adding Subheader for Text Loader
        2. Code Section: Example of loading a text document
        3. Code Explanation: Explanation of how TextLoader works
        """
        st.subheader("Text Loader")
        st.code("""
                    from langchain_community.document_loaders import TextLoader

                    # Loading the text document from a file named 'speech.txt'
                    loader = TextLoader("speech.txt")
                    text_document = loader.load()

                    text_document
                """)
        # Explanation of TextLoader functionality
        st.info(
            "Explanation: The `TextLoader` class is used to load text files as documents that can be processed by LangChain.")

    # PDF Loader Tab
    with tab2:
        st.subheader("PDF Loader")
        st.code("""
                    from langchain_community.document_loaders import PyPDFLoader

                    # Loading a PDF document from a file named 'attention.pdf'
                    loader = PyPDFLoader("attention.pdf")
                    pdf_document = loader.load()

                    pdf_document
                """)
        # Explanation of PDFLoader functionality
        st.info(
            "Explanation: The `PyPDFLoader` is used to load PDF files and process them as documents for further use.")

    # Web Loader Tab
    with tab3:
        st.subheader("Web Loader")
        st.code("""
                    from langchain_community.document_loaders import WebBaseLoader
                    import bs4

                    # Loading a webpage with the WebBaseLoader
                    loader = WebBaseLoader(
                        web_paths=("https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/",),
                        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                            class_=("post-title", "post-description", "post-content", "post-header")
                        ))
                    )
                    wed_document = loader.load()

                    wed_document
                """)
        # Explanation of WebBaseLoader functionality
        st.info(
            "Explanation: The `WebBaseLoader` is used to load webpage content, with BeautifulSoup providing custom filters for parsing HTML elements.")

    # ArXiv Loader Tab
    with tab4:
        st.subheader("ArXiv Loader")
        st.code("""
                    from langchain_community.document_loaders import ArxivLoader

                    # Loading a paper from ArXiv using its identifier (query)
                    loader = ArxivLoader(query="1605.08386", load_max_doc=2)
                    paper_document = loader.load()

                    paper_document
                """)
        # Explanation of ArXivLoader functionality
        st.info(
            "Explanation: The `ArxivLoader` allows you to load academic papers from ArXiv based on a specific query or paper identifier.")

    # Wikipedia Loader Tab
    with tab5:
        st.subheader("Wikipedia Loader")
        st.code("""
                    from langchain_community.document_loaders import WikipediaLoader

                    # Loading content from a Wikipedia article about Python programming language
                    loader = WikipediaLoader(query="Python programming language")
                    info_document = loader.load()

                    info_document
                """)
        # Explanation of WikipediaLoader functionality
        st.info(
            "Explanation: The `WikipediaLoader` is used to load Wikipedia articles by specifying a query for the article's topic.")

    # Adding a new subheader for the next section: Text Splitting Techniques
    st.subheader("Text Splitting Techniques")

    # Creating tabs for different types of text splitting and loaders
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Recursive Character Text Splitter", "Character Text Splitter", "HTML Header Text Splitter", "JSON Text Splitter"])

    # Recursive Character Text Splitter
    with tab1:
        st.subheader("Recursive Character Text Splitter: Create Documents")
        st.code("""
                    # Loading a speech text file into a string variable
                    speech = ""

                    with open("speech.txt") as file:
                        speech = file.read()

                    # Using RecursiveCharacterTextSplitter to split the text into documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
                    final_speech_document = text_splitter.create_documents([speech])

                    final_speech_document
                """)
        # Explanation of RecursiveCharacterTextSplitter
        st.info(
            r"Explanation: The `RecursiveCharacterTextSplitter` splits long text into smaller chunks on the basis of characters list ['\n\n','\n',''], useful for handling large documents.")

        st.subheader("Recursive Character Text Splitter: Split Documents")
        st.code("""
                            from langchain_text_splitters import RecursiveCharacterTextSplitter

                            # Using RecursiveCharacterTextSplitter to split an existing document (info_document)
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            # Assuming info_document is already loaded, we split it into smaller parts
                            final_info_document = text_splitter.split_documents(info_document)

                            final_info_document
                        """)
        # Explanation of splitting functionality
        st.info(
            r"Explanation: The `split_documents` method breaks the document into chunks on the basis of characters list ['\n\n', '\n', ''], defined by the chunk size and overlap parameters."
        )

    # Character Text Splitter
    with tab2:
        st.subheader("Text Loader and Character Text Splitter: Create Documents")
        st.code("""
                    from langchain_community.document_loaders import TextLoader

                    # Loading the text file into documents using TextLoader
                    text_loader = TextLoader('speech.txt')
                    text_docs = text_loader.load()

                    # Using CharacterTextSplitter to split the loaded documents
                    from langchain_text_splitters import CharacterTextSplitter

                    text_splitter = CharacterTextSplitter(separator="\\n\\n", chunk_size=100, chunk_overlap=20)
                    final_speech_document = text_splitter.split_documents(text_docs)

                    final_speech_document
                """)
        # Explanation of CharacterTextSplitter with TextLoader
        st.info(
            r"Explanation: The `TextLoader` loads the speech text as documents, and the `CharacterTextSplitter` "
            r"splits these documents into smaller chunks based on the specified separator, chunk size, and overlap."
        )

        st.subheader("Character Text Splitter: Split Documents from String")
        st.code("""
                    # Loading speech text directly from a file into a string
                    speech = ""

                    with open("speech.txt") as file:
                        speech = file.read()

                    # Splitting the speech text into documents using CharacterTextSplitter
                    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
                    final_speech_document = text_splitter.create_documents([speech])

                    final_speech_document
                """)
        # Explanation of splitting text directly from a string
        st.info(
            r"Explanation: This example reads the speech text into a string and splits it using `CharacterTextSplitter`, "
            r"with the default separator being '\\n\\n'. You can adjust the chunk size and overlap as needed."
        )

    # HTML Header Text Splitter
    with tab3:
        st.subheader("HTML Header Text Splitter: Split HTML String")
        st.code("""
                    from langchain_text_splitters import HTMLHeaderTextSplitter

                    # HTML string to split
                    html_str = \"\"\" 
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>20 Line HTML Example Without Styles</title>
                    </head>
                    <body>
                        <header>
                            <h1>Welcome to My Website</h1>
                            <nav>
                                <ul>
                                    <li><a href="#home">Home</a></li>
                                    <li><a href="#about">About</a></li>
                                    <li><a href="#contact">Contact</a></li>
                                </ul>
                            </nav>
                        </header>
                        <main>
                            <section id="about">
                                <h2>About Us</h2>
                                <p>This is a sample paragraph about our company or website.</p>
                                <p>We provide high-quality services to our customers.</p>
                            </section>
                            <section id="contact">
                                <h2>Contact Information</h2>
                                <p>Email: info@example.com</p>
                                <p>Phone: (123) 456-7890</p>
                            </section>
                        </main>
                        <footer>
                            <p>&copy; 2024 My Website. All rights reserved.</p>
                        </footer>
                    </body>
                    </html>
                    \"\"\"

                    # Splitting the HTML string based on header tags
                    headers_to_split_on = [
                        ("h1", "header 1"),
                        ("h2", "header 2")
                    ]

                    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
                    final_html_text = html_splitter.split_text(html_str)

                    final_html_text
                """)
        # Explanation of HTMLHeaderTextSplitter
        st.info(
            r"Explanation: The `HTMLHeaderTextSplitter` splits HTML content into smaller chunks based on header tags like "
            r"`<h1>` and `<h2>`. The split is customized based on which headers you want to use as breakpoints."
        )

        st.subheader("HTML Header Text Splitter: Split HTML from URL")
        st.code("""
                    # Splitting HTML content from a URL based on header tags
                    url = "https://python.langchain.com/v0.2/docs/introduction/"

                    headers_to_split_on = [
                        ("h1", "header 1"),
                        ("h2", "header 2"),
                        ("h3", "header 3"),
                        ("h4", "header 4")
                    ]

                    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
                    final_html_text = html_splitter.split_text_from_url(url)

                    final_html_text
                """)
        # Explanation of splitting HTML from a URL
        st.info(
            r"Explanation: This example demonstrates how to use `HTMLHeaderTextSplitter` to split HTML content fetched "
            r"directly from a URL. The content is split based on various header tags (`<h1>`, `<h2>`, etc.) defined by the user."
        )

    # JSON Splitter
    with tab4:
        st.subheader("Recursive JSON Splitter: Split JSON Data")
        st.code("""
                    import json
                    import requests

                    # Fetching JSON data from an API
                    json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

                    # Splitting the JSON data into smaller chunks using RecursiveJsonSplitter
                    from langchain_text_splitters import RecursiveJsonSplitter
                    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
                    json_chunk = json_splitter.split_json(json_data)

                    json_chunk
                """)
        # Explanation of RecursiveJsonSplitter for JSON data
        st.info(
            r"Explanation: The `RecursiveJsonSplitter` splits large JSON objects into smaller chunks, which is useful for processing large amounts of structured data. The `split_json` method is used to divide the data into manageable pieces based on the maximum chunk size."
        )

        st.subheader("Recursive JSON Splitter: Create Documents from JSON Data")
        st.code("""
                    import json
                    import requests

                    # Fetching JSON data from an API
                    json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

                    # Creating documents from JSON data using RecursiveJsonSplitter
                    from langchain_text_splitters import RecursiveJsonSplitter
                    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
                    json_chunk = json_splitter.create_documents(texts=[json_data])

                    json_chunk
                """)
        # Explanation of creating documents from JSON data
        st.info(
            r"Explanation: The `create_documents` method allows you to convert JSON data into documents for further processing, splitting the content into smaller parts based on a maximum chunk size."
        )

    st.subheader("Text Embedding and Vector Storing Techniques")

    # Creating tabs for different types of embeddings and vector storing techniques
    tab1, tab2, tab3 = st.tabs(
        ["OpenAI Embedding", "Ollama Embedding", "Huggingface Embedding"])

    # OpenAI Embedding with ChromaDB Tab
    with tab1:
        """
        1. Adding Subheader for OpenAI Embedding with ChromaDB
        2. Code Section: Example of how to embed text using OpenAI embeddings and store it in ChromaDB
        3. Code Explanation: Explanation of how OpenAIEmbeddings and Chroma work together
        """
        st.subheader("OpenAI Embedding with ChromaDB")
        st.code("""
                    from langchain_text_splitters import RecursiveCharacterTextSplitter

                    # Loading a speech text file into a string variable
                    speech = ""

                    with open("speech.txt") as file:
                        speech = file.read()

                    # Using RecursiveCharacterTextSplitter to split the text into documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
                    final_speech_document = text_splitter.create_documents([speech])

                    # OpenAI Embedding
                    import os 
                    from dotenv import load_dotenv
                    load_dotenv()

                    os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

                    # Converting text into vector
                    from langchain_openai import OpenAIEmbeddings
                    from langchain_community.vectorstores import Chroma

                    openai_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

                    db = Chroma.from_documents(
                        final_speech_document,
                        openai_embedding,
                        collection_name="speech_collection_openai_3072"  # Ensure unique name or reinitialize
                    )

                    query = "Our model achieves 28.4 BLEU"
                    retrived_result = db.similarity_search(query)
                    print(retrived_result)
                """)
        # Explanation of OpenAI embedding with ChromaDB
        st.info(
            "Explanation: The `OpenAIEmbeddings` class is used to convert text into embeddings, "
            "and `ChromaDB` is used as a vector store to perform similarity search.")

    # Ollama Embedding with FAISS Tab
    with tab2:
        """
        1. Adding Subheader for Ollama Embedding with FAISS
        2. Code Section: Example of how to embed PDF text using Ollama embeddings and store it in FAISS Vectorstore
        3. Code Explanation: Explanation of OllamaEmbeddings and FAISS for vector search
        """
        st.subheader("Ollama Embedding with FAISS Vectorstore DB")
        st.code("""
                    from langchain_community.document_loaders import PyPDFLoader
                    from langchain_community.vectorstores import FAISS
                    from langchain_community.embeddings import OllamaEmbeddings
                    from langchain_text_splitters import CharacterTextSplitter
                    
                    # Loading a PDF file into a string variable
                    loader = PyPDFLoader("attention.pdf")
                    documents = loader.load()
                    
                    # Splitting the documents into smaller chunks
                    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30)
                    docs = text_splitter.split_documents(documents)
                    
                    # Ollama Embedding
                    ollama_embedding = (
                        OllamaEmbeddings(model="nomic-embed-text")  # Bydefauilt it will use llama2 model in backend
                    )

                    # Converting text into vector
                    db = FAISS.from_documents(
                        docs,
                        huggingface_embedding
                    )

                    query = "decoder is also composed of a stack of N = 6 identical layers"
                    retrived_result = db.similarity_search(query)
                    print(retrived_result[0].page_content)
                """)
        # Explanation of Ollama embedding with FAISS
        st.info(
            "Explanation: The `OllamaEmbeddings` class is used to embed documents, and `FAISS` is utilized "
            "for storing embeddings and performing similarity search.")

    # Huggingface Embedding with ChromaDB Tab
    with tab3:
        """
        1. Adding Subheader for Huggingface Embedding with ChromaDB
        2. Code Section: Example of embedding text using Huggingface embeddings and storing in ChromaDB
        3. Code Explanation: Explanation of HuggingfaceEmbeddings and Chroma for vector search
        """
        st.subheader("Huggingface Embedding with Chroma DB")
        st.code("""
                    from langchain_text_splitters import RecursiveCharacterTextSplitter

                    # Loading a speech text file into a string variable
                    speech = ""

                    with open("speech.txt") as file:
                        speech = file.read()

                    # Using RecursiveCharacterTextSplitter to split the text into documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
                    final_speech_document = text_splitter.create_documents([speech])

                    # Huggingface Embedding 
                    import os 
                    from dotenv import load_dotenv
                    load_dotenv()

                    os.environ["HUGGINGFACE_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")

                    from langchain_huggingface import HuggingFaceEmbeddings
                    from langchain_community.vectorstores import Chroma

                    # Sentence Transformer on HuggingFaceEmbeddings
                    huggingface_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                    db = Chroma.from_documents(
                        final_speech_document,
                        huggingface_embedding,
                        collection_name="speech_collection_huggingface_3072"  # Ensure unique name or reinitialize
                    )

                    query = "Our model achieves 28.4 BLEU"
                    retrived_result = db.similarity_search(query)
                    print(retrived_result)
                """)
        # Explanation of Huggingface embedding with ChromaDB
        st.info(
            "Explanation: The `HuggingFaceEmbeddings` class converts text into embeddings using Huggingface models, "
            "and `ChromaDB` serves as the vector store for similarity search.")
