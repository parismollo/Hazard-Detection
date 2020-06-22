'''
NOT USED IN THE DEPLOYED VERSION!
'''




# import streamlit as st
# from PIL import Image
# import matplotlib.gridspec as gridspec
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2
# import os
# from random import randint
# from app_demo import fire_class_names, scene_class_names, cifar_class_names, pedestrian_class_names

# def run_performance():
#     st.header('Convolutional Neural Networks')
#     st.info(""" 
#     This section aims to inform the **models architecture**
#     used in the project and their **performance on a test set**
#     """)
#     model = st.radio("Pick one model to evaluate",
#         ('Fire', 'CIFAR10', 'Scene', 'People'))
#     plot_info(model)

# def show_image(path, caption='Network Architecture'):
#     image = Image.open(path)
#     st.image(image, caption=caption, use_column_width=False)

# def plot_info(model_name):
#     if model_name == 'Fire':
#         st.subheader('Fire')
#         show_image('images/networks_arch/fire_model.png')
#         run_evaluation((10, 10), (150,150), 'models/fire.h5', 2,fire_class_names)

#     elif model_name == 'CIFAR10':
#         st.subheader('CIFAR10')
#         show_image('images/networks_arch/cifar_model.png')
#         run_evaluation((15, 15), (32, 32), 'models/cifar.h5', 10, cifar_class_names)

#     elif model_name == 'Scene':
#         st.subheader('Scene')
#         show_image('images/networks_arch/scene_model.png')
#         run_evaluation((10, 10), (150, 150), 'models/scene.h5', 6, scene_class_names)
    
#     elif model_name == 'People':
#         st.subheader('People')
#         # show_image('images/networks_arch/people.png')
#         run_evaluation((10, 10), (150, 150), 'models/pedestrian.h5', 2, pedestrian_class_names)

# def get_images(shape: tuple):
#     Images = []
#     for image_file in os.listdir('images/performane_test_data'):
#         image = cv2.imread('images/performane_test_data/'+image_file) 
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image,shape) #Resize the image, Some images are different sizes. (Resizing is very Important)
#         Images.append(image)
#     return Images

# def plot_predictions(figsize, pred_images, model, reshape, classes):

#     fig = plt.figure(figsize=figsize)
#     outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

#     for i in range(4):
#         inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
#         pred_image = np.array([pred_images[i]])
#         pred_prob = model.predict(pred_image).reshape(reshape)
#         for j in range(2):
#             if (j%2) == 0:
#                 ax = plt.Subplot(fig, inner[j])
#                 ax.imshow(pred_image[0].astype('uint8'))
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 fig.add_subplot(ax)
#             else:
#                 ax = plt.Subplot(fig, inner[j])
#                 ax.bar(classes,pred_prob)
#                 fig.add_subplot(ax)


#     st.pyplot()


# def run_evaluation(figsize, shape, model_path, reshape, classes):
#     pred_images = get_images(shape)
#     pred_images = np.array(pred_images).astype(np.float32)
#     model = tf.keras.models.load_model(model_path)
#     plot_predictions(figsize, pred_images, model, reshape, classes)
