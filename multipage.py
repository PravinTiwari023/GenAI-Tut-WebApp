import streamlit as st
class MultiPage:
    def __init__(self):
        self.pages = []

    def add_page(self, title, func):
        """Adds a new page to the app."""
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        page = st.sidebar.selectbox(
            'Select a Page:',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()
