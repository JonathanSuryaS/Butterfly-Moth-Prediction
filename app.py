#Import Libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

#Load model
model = load_model(r"C:\Users\user\Documents\Datasets\Butterfly & Moths\butterly_moth.h5")

#Name of Classes
CLASS_NAMES = ['ADONIS',
 'AFRICAN GIANT SWALLOWTAIL',
 'AMERICAN SNOOT',
 'AN 88',
 'APPOLLO',
 'ARCIGERA FLOWER MOTH',
 'ATALA',
 'ATLAS MOTH',
 'BANDED ORANGE HELICONIAN',
 'BANDED PEACOCK',
 'BANDED TIGER MOTH',
 'BECKERS WHITE',
 'BIRD CHERRY ERMINE MOTH',
 'BLACK HAIRSTREAK',
 'BLUE MORPHO',
 'BLUE SPOTTED CROW',
 'BROOKES BIRDWING',
 'BROWN ARGUS',
 'BROWN SIPROETA',
 'CABBAGE WHITE',
 'CAIRNS BIRDWING',
 'CHALK HILL BLUE',
 'CHECQUERED SKIPPER',
 'CHESTNUT',
 'CINNABAR MOTH',
 'CLEARWING MOTH',
 'CLEOPATRA',
 'CLODIUS PARNASSIAN',
 'CLOUDED SULPHUR',
 'COMET MOTH',
 'COMMON BANDED AWL',
 'COMMON WOOD-NYMPH',
 'COPPER TAIL',
 'CRECENT',
 'CRIMSON PATCH',
 'DANAID EGGFLY',
 'EASTERN COMA',
 'EASTERN DAPPLE WHITE',
 'EASTERN PINE ELFIN',
 'ELBOWED PIERROT',
 'EMPEROR GUM MOTH',
 'GARDEN TIGER MOTH',
 'GIANT LEOPARD MOTH',
 'GLITTERING SAPPHIRE',
 'GOLD BANDED',
 'GREAT EGGFLY',
 'GREAT JAY',
 'GREEN CELLED CATTLEHEART',
 'GREEN HAIRSTREAK',
 'GREY HAIRSTREAK',
 'HERCULES MOTH',
 'HUMMING BIRD HAWK MOTH',
 'INDRA SWALLOW',
 'IO MOTH',
 'Iphiclus sister',
 'JULIA',
 'LARGE MARBLE',
 'LUNA MOTH',
 'MADAGASCAN SUNSET MOTH',
 'MALACHITE',
 'MANGROVE SKIPPER',
 'MESTRA',
 'METALMARK',
 'MILBERTS TORTOISESHELL',
 'MONARCH',
 'MOURNING CLOAK',
 'OLEANDER HAWK MOTH',
 'ORANGE OAKLEAF',
 'ORANGE TIP',
 'ORCHARD SWALLOW',
 'PAINTED LADY',
 'PAPER KITE',
 'PEACOCK',
 'PINE WHITE',
 'PIPEVINE SWALLOW',
 'POLYPHEMUS MOTH',
 'POPINJAY',
 'PURPLE HAIRSTREAK',
 'PURPLISH COPPER',
 'QUESTION MARK',
 'RED ADMIRAL',
 'RED CRACKER',
 'RED POSTMAN',
 'RED SPOTTED PURPLE',
 'ROSY MAPLE MOTH',
 'SCARCE SWALLOW',
 'SILVER SPOT SKIPPER',
 'SIXSPOT BURNET MOTH',
 'SLEEPY ORANGE',
 'SOOTYWING',
 'SOUTHERN DOGFACE',
 'STRAITED QUEEN',
 'TROPICAL LEAFWING',
 'TWO BARRED FLASHER',
 'ULYSES',
 'VICEROY',
 'WHITE LINED SPHINX MOTH',
 'WOOD SATYR',
 'YELLOW SWALLOW TAIL',
 'ZEBRA LONG WING']

#App Title
st.title('Butterfly & Moth Prediction')
st.markdown('Upload an image of butterfly / moths')

#Upload section
image = st.file_uploader('Please upload an image', type = 'jpg')
submit = st.button('Predict')


if submit:
    if image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        #Convert BGR to RGB
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Displaying the image
        st.image(opencv_image, channels="RGB")
        st.write(opencv_image.shape)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        
        #Normalize Image
        opencv_image = opencv_image.astype('float32') / 255.0
        
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(f'This is {result}')