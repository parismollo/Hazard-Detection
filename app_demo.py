import streamlit as st
from preprocessing.preprocessing import pre_cifar, pre_scene, pre_fire, pre_pedestrian
import tensorflow as tf
from tensorflow.keras import models, layers
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from filters_feature import plot_filters, plot_feature_maps

cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
scene_class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
fire_class_names = ['no fire', 'fire']
pedestrian_class_names = ['no people', 'people']


fire_model_path = ('Fire', 'models/fire.h5')
cifar_model_path = ('CIFAR10', 'models/cifar.h5')
scene_model_path = ('Scene', 'models/scene.h5')
pedestrian_model_path = ('People', 'models/pedestrian.h5')


fire_shape = (150, 150)
cifar_shape = (32, 32)
scene_shape = (150, 150)
pedestrian_shape = (150, 150)



def load_image():
    uploaded_file = st.file_uploader("Choose a jpg file", type=["jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.save('images/file.jpg')
        return True

def plot_input():
    image = cv2.imread('images/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.title('User input')
    st.pyplot()

def cifar():
    image = pre_cifar()

    with st.spinner('Loading model...'):
        model = tf.keras.models.load_model('models/cifar.h5')
    

    pred_prob = model.predict(image).reshape(10)
    prob = pd.DataFrame(pred_prob, columns=['confidence'], index=cifar_class_names)
    prob_max = prob.style.highlight_max(axis=0)

    return prob, prob_max
    
def scene():
    image = pre_scene()
    
    with st.spinner('Loading model...'):
        model = tf.keras.models.load_model('models/scene.h5')
    
    pred_prob = model.predict(image).reshape(6)
    prob = pd.DataFrame(pred_prob, columns=['confidence'], index=scene_class_names)
    prob_max = prob.style.highlight_max(axis=0)

    return prob, prob_max

def pedestrian():
    image = pre_pedestrian()
    
    with st.spinner('Loading model...'):
        model = tf.keras.models.load_model('models/pedestrian.h5')
    
    pred_prob = model.predict(image).reshape(2)
    prob = pd.DataFrame(pred_prob, columns=['confidence'], index=pedestrian_class_names)
    prob_max = prob.style.highlight_max(axis=0)

    return prob, prob_max



def fire():
    image = pre_fire()
    
    with st.spinner('Loading model...'):
        model = tf.keras.models.load_model('models/fire.h5')
    
    pred_prob = model.predict(image).reshape(2)
    prob = pd.DataFrame(pred_prob, columns=['confidence'], index=fire_class_names)
    prob_max = prob.style.highlight_max(axis=0)

    return prob, prob_max


def run_demo():
    st.title('HEPS')
    st.header('Description')
    st.info("""
    The initial concept of this project implied building a web application that could 
    identify fire hazards on images and output a description 
    of the situation portrayed in the image (e.g. environment 
    classification, objects classification, pedestrian detection)
    """)
    st.markdown("""
    This framework could substantially improve the efficiency of fire 
    fighting in smart cities, it reduces the 
    time to detect hazardous situations, hence it 
    speeds up the fire department time to act.
    """)


    st.header('How it works?')
    st.error("""This is an **Alpha version** of the project, real world applications would require a video processing 
    that would feed the machine learning models API's, instead of an user input image presented in this demo.""")
    st.success("""
    Pick an *.jpg/.jpeg* image of your choice, save the image and load on the section below. 
    The models' labels are displayed on the sidebar menu, 
    once you select the model you desire to run, the 
    predictions will appear below each label.
    """)

    st.header('Try it out!')
    path = load_image()
    if path is not None:

        plot_input()
        st.sidebar.title('Models')
        run_fire = st.sidebar.checkbox('Fire')
        if run_fire:
            prob_f, prob_max_f = fire()
            st.sidebar.table(prob_max_f)

        run_scene = st.sidebar.checkbox('Environment')
        if run_scene:
            prob_s, prob_max_s = scene()
            st.sidebar.table(prob_max_s)

        run_cifar = st.sidebar.checkbox('CIFAR10')
        if run_cifar:
            st.warning("Attention! If image doesn't contains CIFAR10 objects, model will not be accurate.")
            prob_c, prob_max_c = cifar()
            st.sidebar.table(prob_max_c)
        
        run_pedestrian = st.sidebar.checkbox('People')
        if run_pedestrian:
            prob_c, prob_max_c = pedestrian()
            st.sidebar.table(prob_max_c)

        

def run_how_it_works():
    st.subheader('Convolutional Filters and Feature maps')
    st.info("""
        The first plot represents a set of **filters from the 
        fire model** while the second plot represents a **set 
        of feature maps** from the model specified below
    """)
    plot_filters('models/fire.h5')

    path = st.radio("Pick one model to plot the Feature maps",
    ('Fire', 'CIFAR10', 'Scene', 'People'))
    layer_number = st.number_input('Choose a layer number')
    layer_number = int(layer_number)
    if path == 'Fire':
        model_path = fire_model_path[-1]
        shape = fire_shape
    elif path == 'CIFAR10':
        model_path = cifar_model_path[-1]
        shape = cifar_shape
    elif path == 'Scene':
        model_path = scene_model_path[-1]
        shape = scene_shape
    elif path == 'People':
        model_path = pedestrian_model_path[-1]
        shape = pedestrian_shape
    plot_feature_maps(model_path, layer_number, shape)
