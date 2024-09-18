import streamlit as st


def app():
    # Title and Introduction
    st.title("Generative AI and LangChain Tutorial")
    st.write("""
    Welcome to the **Generative AI and LangChain** tutorial cheatsheet! 
    This app serves as a comprehensive guide to understanding and implementing **Generative AI** using tools like **LangChain**. 

    Whether you're a beginner or an expert, this platform will help you explore the evolving world of AI with practical examples, code snippets, and explanations.
    """)

    # Section: What is Generative AI?
    st.header("What is Generative AI?")
    st.write("""
    Generative AI refers to algorithms that can create new data, such as images, text, or code, from a given dataset. Popular examples include OpenAI's GPT series, DALL·E, and diffusion models.

    With generative models, you can produce realistic content, automate processes, and improve efficiency in various fields, from creative arts to technical industries.
    """)

    # Section: What is LangChain?
    st.header("What is LangChain?")
    st.image("https://python.langchain.com/v0.2/svg/langchain_stack_062024.svg", caption="LangChain simplifies every stage of the LLM application lifecycle")
    st.write("""
    **LangChain** is a framework that simplifies building applications that leverage large language models (LLMs) like GPT. It allows you to connect LLMs with external data, tools, and services to build sophisticated, real-world applications.

    LangChain supports components like prompt templates, memory management, document loading, and more, making it easier to build AI applications that can interact with external data sources.
    """)

    # Navigation Instructions
    st.subheader("How to Use This App")
    st.write("""
    Use the **sidebar** to navigate through different sections of this app. Each section provides code examples, explanations, and tutorials on various topics related to Generative AI and LangChain.

    The available sections include:
    - **LangChain Components**: Dive into different LangChain modules.
    - **Generative AI Models**: Explore how to use and build with generative AI.
    """)

    # Final Note
    st.write("""
    Ready to explore? Select a section from the sidebar and start your journey into the world of Generative AI and LangChain!
    """)
