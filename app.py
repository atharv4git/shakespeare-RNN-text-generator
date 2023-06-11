from model import model_output
import streamlit as st
import streamlit.components.v1 as components
from markdown_it import MarkdownIt

st.title("Shakespeare AI writings")
st.image("s1.jpeg")


def show_popup():
    with open("about.md", "r") as file:
        content = file.read()
    st.markdown(content)


if st.button("Learn abour RNNs"):
    with st.expander("Markdown Content", expanded=True):
        show_popup()

num = int(st.number_input("Enter the number of lines to generate"))
if num:
    st.title("RNN generated output")
    st.write(model_output(num*100))
