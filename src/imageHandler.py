from PIL import Image, ImageDraw
import math
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random



def drawBoundingRectangle(coordsTup , ImgObj, color='#ff0000', width=5, alpha=1.0):
    """
    Creates a box to highligt a location on the image object

    coordsTup = tuple of bbox corners. 
        top left coords = first 2 elements
        bottom right coords = last 2 elements
    width: width of outline
    color: hex value of color. Default red
    alpha: opacity. Range: 0,1
    Note: jpeg does not support alpha
    """
    #Checking for transparency
    if alpha > 1 or alpha < 0: 
        alpha =1
    
    color_with_opacity = color + hex(int(alpha*255))[-2:]
    
    # Draw a rectangle
    draw = ImageDraw.Draw(ImgObj,'RGBA')
    
    p1_coord = coordsTup[0:2]
    p2_coord = coordsTup[2:4]
    draw.rectangle([p1_coord, p2_coord], outline=color_with_opacity, width=width)
    
    
def createNewPotentialBboxes(imgObj):
    '''
    Returns a dataframe that contains the bounding box coordinates and window size
    for all potential bounding boxes as defined by the size of the image and 
    a hard-coded set of window sizes

    Dataframe is also saved as a CSV file is saved in the same directory as the large maps
    '''
    img_width,  img_height = imgObj.size
    overlap = 0.75

    df_list = []
    

    for winsize in [(277*0.5), (277*0.75) , (277), (277*1.5),(277*2),(277*2.5),(277*3)]:
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
    '''
    x_cnt : x coordinate of center of window
    y_cnt : y coordinate of center of window
    winsize: Size of window (bounding box)

    Returns a tuple of (x-min, y-min, x-max, y-max) of the bounding box
    '''
    return int(x_cnt - winsize/2),int(y_cnt - winsize/2),int(x_cnt + winsize/2),int(y_cnt + winsize/2)

def getPotentialBboxes(imgObj,forceCreateNew = False):
    '''
    Checks to see if a file containing bounding boxes already exists for the size of the image passed
    as an argument. 
    If already exists, the csv is loaded, if not a fresh file is created.

    forceCreateNew =    ignores check of file and creates (and overwrites) csv file of potential 
                        bounding boxes
    '''
    img_width,  img_height = imgObj.size
    
    if forceCreateNew:
        print('force creating new')
        df_bboxPotentials = createNewPotentialBboxes(imgObj)
    else:
        try:
            print('reading current')
            df_bboxPotentials = pd.read_csv(f'mass_maps/bboxPotentials_{img_width}_{img_height}.csv',
            converters={"bbox_bounds": eval}
            )
        except FileNotFoundError:
            print('no existing doc. creating')
            df_bboxPotentials = createNewPotentialBboxes(imgObj)
        
        
    return df_bboxPotentials

def getClassFromPred(pred,threshold):
    '''
    pred : predicted probability of all classes
    threshold : predicted probability threshold that needs to be met to be considered a valid prediction

    Returns the predicted class 
    '''
    if max(pred)>threshold :
        return pred.argmax()
    return -1

def saveImageSlices(listOfImages,fName):
    '''
    Saves list of images to a specific folder
    Recreates folder on each run.
    listOfImages : list of images to save
    fName : used in the naming convention of the slices
    '''
    slicePath = f"./mass_maps/_slices_img/"
    
    if (os.path.exists(slicePath) and os.path.isdir(slicePath)):
        shutil.rmtree(slicePath)
    os.makedirs(slicePath)

    for count, img in enumerate(listOfImages):
        img.save(''.join([slicePath,str(count),'_',(fName.split('/')[-1])]))