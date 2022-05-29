import streamlit as st
import tensorflow as tf
from glob import glob
import random
from PIL import Image, ImageDraw

import numpy as np
np.random.seed(42)

# def drawBoundingRectangle(coordsTup , ImgObj, color='#ff0000', width=5, alpha=1.0):
#     """
#     Creates a box to highligt a location on the image object

#     coordsTup = tuple of bbox corners. 
#         top left coords = first 2 elements
#         bottom right coords = last 2 elements
#     width: width of outline
#     color: hex value of color. Default red
#     alpha: opacity. Range: 0,1
#     """
#     #Checking for transparency
#     if alpha > 1 or alpha < 0: 
#         alpha =1
    
#     color_with_opacity = color + hex(int(alpha*255))[-2:]
    
#     # Draw a rectangle
#     draw = ImageDraw.Draw(ImgObj,'RGBA')
    
#     p1_coord = coordsTup[0:2]
#     p2_coord = coordsTup[2:4]
#     draw.rectangle([p1_coord, p2_coord], outline=color_with_opacity, width=width)
    
    
# def createNewPotentialBboxes(imgObj):
#     img_width,  img_height = imgObj.size
#     overlap = 0.4
#     stride = 277*(1-overlap)

#     df_list = []
    
# #     (277,277*1.25,277*1.5)
#     for winsize in [ (277*0.5) , (277), (277*1.25)]:
#         winsize = round(winsize)
#         if (winsize > img_width) or (winsize > img_height):
#             break
            
            
#         #if winsize odd, increase by 1 pixel. Just easier
#         if (winsize % 2) != 0:
#             winsize+=1
#         xvals = np.arange((winsize/2), (img_width-winsize/2),stride)
#         yvals = np.arange((winsize/2), (img_height-winsize/2),stride)
#         list_centroids = [(x,y) for x in xvals for y in yvals]

#         list_coordTuples = [get_bboxCoords(x_cnt,y_cnt,winsize) for x_cnt,y_cnt in list_centroids]

#         #Appends new records for current winsize and bbox
#         df_list.append(pd.DataFrame([list_coordTuples,[winsize]*len(list_coordTuples)]).T)

#     df_bboxPotentials = pd.concat(df_list,ignore_index=True,axis=0)
#     df_bboxPotentials.columns = ['bbox_bounds','winsize']
#     df_bboxPotentials.to_csv(f'mass_maps/bboxPotentials_{img_width}_{img_height}.csv',index = False)
#     return df_bboxPotentials

# def get_bboxCoords(x_cnt,y_cnt,winsize):
#     return int(x_cnt - winsize/2),int(y_cnt - winsize/2),int(x_cnt + winsize/2),int(y_cnt + winsize/2)

# def getPotentialBboxes(imgObj,forceCreateNew = False):
    
#     img_width,  img_height = imgObj.size
    
#     if forceCreateNew:
#         print('force creating new')
#         df_bboxPotentials = createNewPotentialBboxes(imgObj)
#     else:
#         try:
#             print('reading current')
#             df_bboxPotentials = pd.read_csv(f'mass_maps/bboxPotentials_{img_width}_{img_height}.csv')
#         except FileNotFoundError:
#             print('no existing doc. creating')
#             df_bboxPotentials = createNewPotentialBboxes(imgObj)
        
        
#     return df_bboxPotentials

# def getClassFromPred(pred,threshold):
#     if max(pred)>threshold and pred.argmax()!=4 :
#         return pred.argmax()
#     return -1


# def getRandomColor():
#     #Shamelessly stolen from one of the references above
#     #Returns a random hex color value
#     return str(["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0])


# colorMap = {n:getRandomColor() for n in range(0,8)}


model = tf.keras.models.load_model('final_finalModel__jk__butReally_FinalModel.h5py')
class_names = {'bright dune': 0,
 'crater': 1,
 'dark dune': 2,
 'impact ejecta': 3,
 'other': 4,
 'slope streak': 5,
 'spider': 6,
 'swiss cheese': 7}

options={}
colorMap={}

st.title("Mars Anomaly Detection")

img_path= None

if img_path is None:
    img_path = st.file_uploader("Choose an image file", type="jpg")

if img_path is not None:
    st.image(img_path)

    st.write('Select features to identify:')
    options['bright dune'] = st.checkbox('Bright Dunes')
    options['crater'] = st.checkbox('Craters')
    options['dark dune'] = st.checkbox('Dark Dunes')
    options['impact ejecta'] = st.checkbox('Impact Ejecta')
    options['slope streak'] = st.checkbox('Slope Streak')
    options['spider'] = st.checkbox('Spider')
    options['swiss cheese'] = st.checkbox('Swiss Cheese')

    options['other'] = st.checkbox('Other')


# st.write(colorMap)
st.write('Pick colors:')

# st.write(class_names['crater'])
# st.write(ColorMap[class_names['crater']])
with st.sidebar:
    
    colorMap[class_names['bright dune']] = st.color_picker('Bright Dune', '#0796FF')
    colorMap[class_names['crater']] = st.color_picker('Crater', '#D626CF')
    colorMap[class_names['dark dune']] = st.color_picker('Dark Dune', '#8941F5')
    colorMap[class_names['impact ejecta']] = st.color_picker('Impact Ejecta', '#46760A')
    colorMap[class_names['slope streak']] = st.color_picker('Slope Streak', '#4C1634')
    colorMap[class_names['spider']] = st.color_picker('Spider', '#C75280')
    colorMap[class_names['swiss cheese']] = st.color_picker('Swiss Cheese', '#481363')
    colorMap[class_names['other']] = st.color_picker('Other', '#F277FD')