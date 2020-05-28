import cv2
import numpy as np

# could develop a single function for both and use arguments for customization

def pre_cifar():
    image = cv2.imread('images/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(32,32))
    image = image / 255.0
    image = np.array(image).astype(np.float32)
    image = (np.expand_dims(image,0))

    return image


def pre_scene():
    image = cv2.imread('images/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(150,150))
    image = np.array(image).astype(np.float32)
    image = (np.expand_dims(image,0))

    return image


def pre_fire():
    image = cv2.imread('images/file.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(150,150))
    image = np.array(image).astype(np.float32)
    image = (np.expand_dims(image,0))

    return image

def pre_pedestrian():
    image = cv2.imread('images/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(150,150))
    image = np.array(image).astype(np.float32)
    image = (np.expand_dims(image,0))

    return image