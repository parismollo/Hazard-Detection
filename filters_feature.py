import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims

def plot_filters(model_path):
    model = tf.keras.models.load_model(model_path)

    filters, biases = model.layers[1].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    st.pyplot()

def plot_feature_maps(model_path, layer_number, shape):
    model = tf.keras.models.load_model(model_path)
    if layer_number < len(model.layers):
        layer = model.layers[layer_number]
        if 'conv' in layer.name:
            compute_feature_map(model, layer_number, shape)
            st.markdown(f"Current layer: **{layer.name}** - *nº {layer_number}*")
        else:
            default_layer_number = 0
            layer = model.layers[default_layer_number]
            st.markdown(f"Current layer: **{layer.name}** - *nº {default_layer_number}*")
            st.error('Try another number for a convolutional layer, **hint**: *check the model architecture on the app*')
            compute_feature_map(model, default_layer_number, shape)
    else:
        st.error(f"""This model has {len(model.layers)-1} layers. 
        This layer number is out of range or not Convolutional. Select a different value in the accepted range.""")
    
         
            
    
def compute_feature_map(model, layer_number, shape):
    model = Model(inputs=model.inputs, outputs=model.layers[layer_number].output)

    model.summary()
    img = load_img('images/file.jpg', target_size=shape)
    img = img_to_array(img)

    img = expand_dims(img, axis=0)
    feature_maps = model.predict(img)

    square = 5
    ix = 1
    for _ in range(square):
        for _ in range(square):

            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    st.pyplot()
