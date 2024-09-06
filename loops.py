import streamlit as st

def app():
    st.title("Loops in Python")

    st.subheader("For Loop")
    st.code("""
    # For loop example
    for i in range(5):
        print(i)
    """)
    st.write("Explanation: Loops help you execute a block of code repeatedly.")
