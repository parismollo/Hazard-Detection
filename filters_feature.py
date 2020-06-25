import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from numpy import expand_dims
from app_demo import fire_model_path, cifar_model_path, scene_model_path, pedestrian_model_path, fire_shape, cifar_shape, pedestrian_shape, scene_shape

def plot_filters(model_path):
    model = tf.keras.models.load_model(model_path)

    filters, biases = model.layers[0].get_weights()
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

def plot_feature_maps(model_path, shape):
    model = tf.keras.models.load_model(model_path)
    layers_dict = {}
    layers_list = []
    i = 0

    for layer in model.layers:
        layers_dict[i] = layer.name
        i+=1

    for n in range(len(model.layers)):
        layers_list.append(layers_dict[n])
    

    # print(layers_dict)
    # print(layers_list)

    layer_name = st.selectbox("Choose the layer",layers_list) 
    
    for number, name in layers_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if name == layer_name:
            layer_number = number
    # print(model.layers)
    layer = model.layers[layer_number]
    st.markdown(f"Selected Layer nº{layer_number} of {len(model.layers)} -  Type:**{layer.name}**")
    if layer_number < len(model.layers):
        layer = model.layers[layer_number]
        if 'conv' in layer.name:
            compute_feature_map(model, layer_number, shape)
            # st.markdown(f"Current layer: **{layer.name}** - *nº {layer_number}*")
        else:
            default_layer_number = 0
            layer = model.layers[default_layer_number]
            # st.markdown(f"Current layer: **{layer.name}** - *nº {default_layer_number}*")
            st.error('Ops...Try another layer! **Hint**: Look for convolutional layers')
            # compute_feature_map(model, default_layer_number, shape)
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

def run_how_it_works():
    st.header('Convolutional Filters and Feature maps')
    st.info("""
        The principal constituent of a convolutional neural network is a 
        **convolutional layer**. Convolutional layers are responsible for 
        the application of "**filters**". Filters identify specific features 
        through a "matrix" operation known as convolution. The result of the 
        convolution will be a feature map (i.e.highlighted features of the input)
    """)
    st.subheader('Filters look like this')
    plot_filters('models/fire_e20_3vgg_zoom_brigh.h5')

 
    st.subheader('Feature Maps')
    st.success("""
        The overall structure of a CNN can be represented as a sequence of 
        convolutional layers, responsible for the identification of features 
        (e.g. lines, circles, shapes, etc ) and pooling layers, 
        responsible to reduce the computational load of the network. 
        **Deeper layers are likely to identify 
        complex shapes, while the low-level layers will identify simpler shapes.** 
    """)
    path = st.selectbox("Select a model",
        ["Fire", "Scene", "People"])
    st.warning("This will take a few seconds...:hourglass_flowing_sand:")
    if path == 'Fire':
        model_path = fire_model_path[-1]
        shape = fire_shape
    # elif path == 'CIFAR10':
    #     model_path = cifar_model_path[-1]
    #     shape = cifar_shape
    elif path == 'Scene':
        model_path = scene_model_path[-1]
        shape = scene_shape
    elif path == 'People':
        model_path = pedestrian_model_path[-1]
        shape = pedestrian_shape

    plot_feature_maps(model_path, shape)
