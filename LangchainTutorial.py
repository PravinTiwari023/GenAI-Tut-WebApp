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
    st.title("Data Loading and Preprocessing with LangChain")

    # Subheader for the first section: Data Ingestion Techniques
    st.subheader("Data Ingestion Techniques")

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
        st.write(
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
        st.write(
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
        st.write(
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
        st.write(
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
        st.write(
            "Explanation: The `WikipediaLoader` is used to load Wikipedia articles by specifying a query for the article's topic.")

    # Adding a new subheader for the next section: Text Splitting Techniques
    st.subheader("Text Splitting Techniques")

    # Creating tabs for different types of text splitting and loaders
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Create Documents", "Split Documents", "Web Loader", "ArXiv Loader", "Wikipedia Loader"])

    # Recursive Character Text Splitter: Create Documents Tab
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
        st.write(
            "Explanation: The `RecursiveCharacterTextSplitter` splits long text into smaller chunks, useful for handling large documents.")

    # Recursive Character Text Splitter: Split Documents Tab
    with tab2:
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
        st.write(
            "Explanation: The `split_documents` method breaks the document into chunks, defined by the chunk size and overlap parameters.")

    # Web Loader Tab for Text Splitting
    with tab3:
        st.subheader("Web Loader")
        st.code("""
                    from langchain_community.document_loaders import WebBaseLoader
                    import bs4

                    # Loading a webpage with the WebBaseLoader and filtering specific HTML elements
                    loader = WebBaseLoader(
                        web_paths=("https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/",),
                        bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                            class_=("post-title", "post-description", "post-content", "post-header")
                        ))
                    )
                    wed_document = loader.load()

                    wed_document
                """)
        # Explanation of WebBaseLoader for web content
        st.write("Explanation: The `WebBaseLoader` loads specific elements from webpages using BeautifulSoup.")

    # ArXiv Loader Tab for Text Splitting
    with tab4:
        st.subheader("ArXiv Loader")
        st.code("""
                    from langchain_community.document_loaders import ArxivLoader

                    # Loading academic papers from ArXiv with a specific query
                    loader = ArxivLoader(query="1605.08386", load_max_doc=2)
                    paper_document = loader.load()

                    paper_document
                """)
        # Explanation of ArXivLoader for academic papers
        st.write("Explanation: The `ArxivLoader` is used to load papers from ArXiv by querying specific identifiers.")

    # Wikipedia Loader Tab for Text Splitting
    with tab5:
        st.subheader("Wikipedia Loader")
        st.code("""
                    from langchain_community.document_loaders import WikipediaLoader

                    # Loading a Wikipedia article on Python programming language
                    loader = WikipediaLoader(query="Python programming language")
                    info_document = loader.load()

                    info_document
                """)
        # Explanation of WikipediaLoader for Wikipedia articles
        st.write("Explanation: The `WikipediaLoader` loads the content of Wikipedia articles based on the given query.")
