import streamlit as st
from app_demo import run_demo
from evaluation import run_performance
from PIL import Image


def main():
    image = Image.open('images/logo.png')
    image.thumbnail((120, 120))
    st.sidebar.image(image)
    st.sidebar.title("What to do?")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Demo", "Models performance"])
    if app_mode == "Demo":
        run_demo()
    elif app_mode == "Models performance":
        run_performance()
        

if __name__ == "__main__":
    main()


# Logo/Icon made by <a href="https://www.flaticon.com/authors/good-ware" title="Good Ware">Good Ware</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a>
