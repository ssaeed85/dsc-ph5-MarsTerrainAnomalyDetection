import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import random
from PIL import Image, ImageDraw
import pandas as pd
import os
import shutil

import numpy as np
np.random.seed(42)

def drawBoundingRectangle(coordsTup , imgObj, color='#ff0000', width=5, alpha=1.0):
    """
    Creates a box to highligt a location on the image object

    coordsTup = tuple of bbox corners. 
        top left coords = first 2 elements
        bottom right coords = last 2 elements
    width: width of outline
    color: hex value of color. Default red
    alpha: opacity. Range: 0,1
    """
    #Checking for transparency
    if alpha > 1 or alpha < 0: 
        alpha =1
    
    color_with_opacity = color + hex(int(alpha*255))[-2:]
    
    # Draw a rectangle
    draw = ImageDraw.Draw(imgObj,'RGBA')
    
    p1_coord = coordsTup[0:2]
    p2_coord = coordsTup[2:4]
    draw.rectangle([p1_coord, p2_coord], outline=color_with_opacity, width=width)
    
    
def createNewPotentialBboxes(imgObj):
    img_width,  img_height = imgObj.size
    overlap = 0.4

    df_list = []
    

    for winsize in [ (277*0.75) , (277), (277*1.5)]:
        winsize = round(winsize)        
        
        if (winsize > img_width) or (winsize > img_height):
            break            
        
        stride = round(winsize*(1-overlap))   
        #if winsize odd, increase by 1 pixel. Just easier
        if (winsize % 2) != 0:
            winsize+=1
        xvals = np.arange((winsize/2), (img_width-winsize/2),stride)
        yvals = np.arange((winsize/2), (img_height-winsize/2),stride)
        list_centroids = [(x,y) for x in xvals for y in yvals]

        list_coordTuples = [get_bboxCoords(x_cnt,y_cnt,winsize) for x_cnt,y_cnt in list_centroids]

        #Appends new records for current winsize and bbox
        df_list.append(pd.DataFrame([list_coordTuples,[winsize]*len(list_coordTuples)]).T)

    df_bboxPotentials = pd.concat(df_list,ignore_index=True,axis=0)
    df_bboxPotentials.columns = ['bbox_bounds','winsize']
    df_bboxPotentials.to_csv(f'mass_maps/bboxPotentials_{img_width}_{img_height}.csv',index = False)
    return df_bboxPotentials

def get_bboxCoords(x_cnt,y_cnt,winsize):
    return int(x_cnt - winsize/2),int(y_cnt - winsize/2),int(x_cnt + winsize/2),int(y_cnt + winsize/2)

def getPotentialBboxes(imgObj,forceCreateNew = False):
    
    img_width,  img_height = imgObj.size
    
    if forceCreateNew:
        print('force creating new')
        df_bboxPotentials = createNewPotentialBboxes(imgObj)
    else:
        try:
            print('reading current')
            df_bboxPotentials = pd.read_csv(f'mass_maps/bboxPotentials_{img_width}_{img_height}.csv')
        except FileNotFoundError:
            print('no existing doc. creating')
            df_bboxPotentials = createNewPotentialBboxes(imgObj)
            
    return df_bboxPotentials

def sliceAndDice(listOfImages):

    fname = imgFile.name
    slicePath = f"./mass_maps/_slices_img/"
    st.write(slicePath)

    if (os.path.exists(slicePath) and os.path.isdir(slicePath)):
        shutil.rmtree(slicePath)
    os.makedirs(slicePath)

    for count, img in enumerate(listOfImages):
        img.save(''.join([slicePath,str(count),'_',(fname.split('/')[-1])]))

def getClassFromPred(pred,threshold):
    if max(pred)>threshold :
        return pred.argmax()
    return -1






# ------------------------------------- #
# ----------------MAIN----------------- #
# ------------------------------------- #


colorMap={}
getPredictionsAndVisualize = False
img_path=None
disableButton=True
im_wAnnotations=None
model = tf.keras.models.load_model('tf_TransferLearning_8class_VGG16.hfpy')
class_names = {'bright dune': 0,
 'crater': 1,
 'dark dune': 2,
 'impact ejecta': 3,
 'other': 4,
 'slope streak': 5,
 'spider': 6,
 'swiss cheese': 7}


with st.sidebar:
    st.header('Mars Anomaly Detection')
    st.write('By [Saad Saeed](https://github.com/ssaeed85/dsc-ph5-MarsTerrainAnomalyDetection)')
    st.title('Select colors for features:')    
    colorMap[class_names['bright dune']] = st.color_picker('Bright Dune', '#0796FF')
    colorMap[class_names['crater']] = st.color_picker('Crater', '#FF00F4')
    colorMap[class_names['dark dune']] = st.color_picker('Dark Dune', '#FF9700')
    colorMap[class_names['impact ejecta']] = st.color_picker('Impact Ejecta', '#8DFF00')
    colorMap[class_names['other']] = st.color_picker('Other', '#FFF301')
    colorMap[class_names['slope streak']] = st.color_picker('Slope Streak', '#FF008B')
    colorMap[class_names['spider']] = st.color_picker('Spider', '#00475C')
    colorMap[class_names['swiss cheese']] = st.color_picker('Swiss Cheese', '#000000')



st.title("Mars Anomaly Detection")
col1, col2 = st.columns(2)

# Image loader
# imgFile = st.file_uploader("Choose an image file", type="jpg")
imgFile = st.file_uploader("Choose an image file", type="jpg")


with col1:
    if imgFile:
        im = Image.open(imgFile)
        st.image(im,caption='Welcome To Maahz')
        disableButton = False
        img_width,  img_height = im.size
        df_bbox = getPotentialBboxes(im,forceCreateNew = True)
        list_imagesToClassify = [im.crop(df_bbox.iloc[rowNum]['bbox_bounds']) 
                            for rowNum in range(0,df_bbox.shape[0])]


# Controls. Disabled until image is loaded
# Multiselect toolbar
options = st.multiselect(
     'Which features would you like to identify',
     ['Crater', 'Bright Dunes', 'Dark Dune', 'Impact Ejecta',
     'Slope Streak','Spider','Swiss Cheese','Other'],
     ['Crater', 'Bright Dunes', 'Dark Dune', 'Impact Ejecta',
     'Slope Streak','Spider','Swiss Cheese'],
     disabled=disableButton
     )
options = [str.lower(x) for x in options]
classes_to_visualize = [v for k,v in class_names.items() if k in options]


# Confidence threshold slider
thresh = st.slider('Confidence Threshold',
value=0.80,
min_value = 0.01, 
max_value= 1.0,
disabled=disableButton)




if st.button('Get Predictions', disabled=disableButton):
    sliceAndDice(list_imagesToClassify)

    predict_datagen = ImageDataGenerator(rescale=1./255)

    predict_generator = predict_datagen.flow_from_directory(
        'mass_maps/',
        target_size = (227,227),
        batch_size=100,
        color_mode='rgb',
        class_mode=None
    )
    predict_generator.reset()

    preds = model.predict(predict_generator)

    df_bbox['predClass'] = [getClassFromPred(pred,thresh) for pred in preds]

    mask_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    df_validBbox = df_bbox[df_bbox['predClass'].isin(classes_to_visualize)][['bbox_bounds','predClass']]

    for idx in range(0, df_validBbox.shape[0]):        
        drawBoundingRectangle(df_validBbox.iloc[idx]['bbox_bounds'],
                            mask_img,
                            color=colorMap[df_validBbox.iloc[idx]['predClass']]
                            )

    im_wAnnotations = im.convert('RGB')

    im_wAnnotations.paste(mask_img,(0,0),mask_img)

with col2:
    if im_wAnnotations is not None:
        st.image(im_wAnnotations)