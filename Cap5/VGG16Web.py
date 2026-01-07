import streamlit as st
import cv2 
import numpy as np
from PIL import Image
from keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16
#APlicacion demo modelo de red neuronal vgg16 como base para el entrenamiento
model = vgg16.VGG16(weights='imagenet')
# tamaño de la imagen 
imageZise = 224
#Creacion de la pagina web con titulos
frameST = st.empty()
st.title("Red Neuronal VGG16")
st.sidebar.markdown("# Red Neuronal VGG16")
#variable para cargar la imagen desde pa lagina web
file_image = st.sidebar.file_uploader("Cargar una imagen", type=["jpg", "png", "jpeg"])
#Instruccion para usar la memoria cache del ordenador
@st.cache(allow_output_mutation=True)
#delaclaracion de la funcion para clasificar la imagen
def vgg16_predict(camFrame, imageSize):
    frame = cv2.resize(camFrame, (imageSize, imageSize))
    numpyImage = img_to_array(frame)
    imageBatch = np.expand_dims(numpyImage, axis=0)
    processedImage = vgg16.preprocess_input(imageBatch.copy())
    predictions = model.predict(processedImage)
    labelVgg = decode_predictions(predictions)
    cv2.putText(camFrame, "VGG16: {}, {:.2F}".format(labelVgg[0][0][1], labelVgg[0][0][2]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return camFrame
#Condicional para ejecutar la predicción de la imagen cargada
if file_image is None:
    st.write("No image file")
else:
    img = Image.open(file_image)
    img = np.asarray(img)[:, :, ::-1].copy()
    st.write("Imagen cargada")
    img = vgg16_predict(img, imageZise)
    img=img[:, :, ::-1]
    st.image(img,use_column_width=True)
    if st.button("Download"):
        im_pil = Image.fromarray(img)
        im_pil.save("output.jpg")
        st.write("Download completed")