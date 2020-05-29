import streamlit as st
from app_demo import run_demo, run_how_it_works
from evaluation import run_performance
from PIL import Image


def main():
    image = Image.open('images/logo.png')
    image.thumbnail((120, 120))
    st.sidebar.image(image)
    st.sidebar.markdown('*Made by* **@parismollo**')
    st.sidebar.title("What to do?")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Demo", "Models performance", "How it works"])
    if app_mode == "Demo":
        run_demo()
    elif app_mode == "Models performance":
        run_performance()
    elif app_mode == "How it works":
        run_how_it_works()
        

if __name__ == "__main__":
    main()


# Logo/Icon made by <a href="https://www.flaticon.com/authors/good-ware" title="Good Ware">Good Ware</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a>
