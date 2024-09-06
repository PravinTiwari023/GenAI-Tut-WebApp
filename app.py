import streamlit as st
from multipage import MultiPage
import home
import LangchainTutorial
import loops

# Create an instance of the app
app = MultiPage()

# Add all the app pages here
app.add_page("Home", home.app)
app.add_page("Langchain Tutorial", LangchainTutorial.app)
app.add_page("Loops", loops.app)

# The main app
app.run()