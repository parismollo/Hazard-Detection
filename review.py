import random
import string
import streamlit as st


emojis = [":tiger2:", ":snake:", ":fish:", ":octopus:", ":snail:", ":dog:", ":bird:", ":flipper:", ":frog:"
    ":beetle:", ":wolf:", ":monkey_face:", ":camel:", ":bee:", ":penguin:", ":koala:",]

def get_random_alphaNumeric_string(stringLength=8):
    digits = string.digits
    return ''.join((random.choice(digits) for i in range(stringLength)))


def review():
    st.title('Review')
    st.info("""
    The models are **still on development**, your feedback can help me to build more accurate models.
    **Please leave a description of what you experienced** and **what you expected**. To submit the review, **press CTRL+ENTER**""")
    results_review = st.text_area('Leave a general review on the results and info on the image you tested', height=1)
    st.warning('On **mobile**, after writing the review, **click over** "Press ctrl+Enter to apply", so that the review can be submited.')
    if  results_review:
        s = get_random_alphaNumeric_string(3)
        f = open('reviews/review.txt', 'a')
        n = random.randrange(0, len(emojis))
        f.write(f"{emojis[n]} **says**: *{results_review}*\n")
        f.close()
    st.title('Previous Reviews')
    read_review()


def read_review():
    f = open("reviews/review.txt", "r")
    for line in f:
        st.markdown(f"{line}")

def temporary_message():
    st.title('Ops....')
    st.error('This feature is under development, comeback later.')
    st.markdown("For while, don't hesitate to contact me via social media at @parismollo to talk about the project.")
