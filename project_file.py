##################################################################################################
import cv2                                 
import numpy  
import pandas as pd                        
import matplotlib.pyplot as plt             
import statistics                           
import time                                 
import datetime
import os, errno                           
import sys
from collections import OrderedDict
import argparse
import mediapipe as mp                  
import skimage
from skimage import exposure
from skimage.transform import resize
import numpy as np
###################################################################################################

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')
## Create Directory
path = 'results/' + st
try:
    os.makedirs(path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
# Create text file
file = open(path + '/results.txt','w')

##########################################################
# Initialize the Mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

################ Provide image path here ###############
img_path='t2.PNG'
##########################################################

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(img_path)
image = cv2.resize(image, (300,300))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output_rgb = None

###############
plt.title('Input Image')
plt.imshow(image)
plt.axis('off')
plt.show()
#############

################################################################################

rgb_image = image
change_background_mp = mp.solutions.selfie_segmentation
change_bg_segment = change_background_mp.SelfieSegmentation()
result = change_bg_segment.process(rgb_image)
binary_mask = result.segmentation_mask > 0.9
binary_mask_np = np.array(binary_mask)
resized_binary_mask = resize(binary_mask_np, (500,550), mode='constant')
resized_binary_mask_gray = (resized_binary_mask * 255).astype(np.uint8)

##################################################################################################

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks_np = np.zeros((468, 2), dtype=np.int32)
        for i, landmark in enumerate(face_landmarks.landmark):
          landmarks_np[i] = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        hull = cv2.convexHull(landmarks_np)
        cv2.fillConvexPoly(mask, hull, 255)
        face_extracted = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

##################################################################################################

        resized_binary_mask_gray = (binary_mask_np * 255).astype(np.uint8)
        # Perform bitwise AND operation between the two masks
        combined_mask = cv2.bitwise_and(resized_binary_mask_gray, mask)
        cv2.imwrite(path + '/ImageFaceFeatures.tif', combined_mask)
        # Apply a bitwise AND operation to the face_extracted image and the combined_mask
        output_rgb = cv2.bitwise_and(face_extracted, face_extracted, mask=combined_mask)
        # Wait for key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

########################################################################################################
#This part of the code will display images 
########################################################################################################

cv2.imwrite(path + '/ImageCrop.tif', output_rgb)
if output_rgb is not None:
    plt.title('Cropped Image')
    plt.imshow(output_rgb)
    plt.axis('off')
    plt.show()

output_gray = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2GRAY)
if output_gray is not None:
    plt.title('Cropped Image in grayscale')
    plt.imshow(output_gray, cmap='gray')
    plt.axis('off')
    plt.show()

###########################################################################################################
# Following matrix provide green and blue coordinates for reference vectors for melanin and hemoglobin and 
#were determined manually in Mathcad routine.
###########################################################################################################

MelArray = []
HemArray = []

Mixed = numpy.matrix([[2, 3], [3, 0]])
UM = numpy.linalg.inv(Mixed)

img = output_rgb

I = img.astype('double')

# Get Red Channel
img0 = I.copy()
img0 = img0[:,:,2]
img0 = img0.astype('double')

# Get Green Channel
img1 = I.copy()
img1 = img1[:,:,1]
img1 = img1.astype('double')

# Get Blue Channel
img2 = I.copy()
img2 = img2[:,:,0]
img2 = img2.astype('double')

tempArray = numpy.sqrt(numpy.square(img0) + numpy.square(img1))
maxV = numpy.amax(tempArray)
numpy.argmax(tempArray)
maxValue = numpy.where(tempArray == maxV)
kr1 = 65534/img0[maxValue][0]
kg1 = 65534/img1[maxValue][0]

tempArray = numpy.sqrt(numpy.square(img0) + numpy.square(img2))
maxV = numpy.amax(tempArray)
numpy.argmax(tempArray)
maxValue = numpy.where(tempArray == maxV)
kb1 = 65534/img2[maxValue][0]
kr2 = 65534/img0[maxValue][0]

tempArray = numpy.sqrt(numpy.square(img1) + numpy.square(img2))
maxV = numpy.amax(tempArray)
numpy.argmax(tempArray)
maxValue = numpy.where(tempArray == maxV)
kg2 = 65534/img1[maxValue][0]
kb2 = 65534/img2[maxValue][0]

# Take mean of constants
kr = (kr1 + kr2)/2
kg = (kg1 + kg2)/2
kb = (kb1 + kb2)/2


# Write adjusters to file
print('kr, kg, kb')
file.write('kr, kg, kb')

print('kr = ',kr)
file.write('\nkr = ' + str(kr))
print('kg = ',kg)
file.write('\nkg = ' + str(kg))
print('kb = ',kb)
file.write('\nkb = ' + str(kb))

##############################################################################################################

img =   cv2.imread(img_path)
img =   cv2.resize(img, (300,300))
img =   cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite(path + '/Image.tif', img)

# NumPy to double
I = img.astype('double')

# Get Red Channel
img0 = I.copy()
img0 = img0[:,:,2]
img0 = img0.astype('double')

# Get Green Channel
img1 = I.copy()
img1 = img1[:,:,1]
img1 = img1.astype('double')

# Get Blue Channel
img2 = I.copy()
img2 = img2[:,:,0]
img2 = img2.astype('double')

# Calculate melanin and hemoglobin images
log_r = -numpy.log10((kr*img0+1)/65535)
log_g = -numpy.log10((kg*img1+1)/65535)
log_b = -numpy.log10((kb*img2+1)/65535)

Melanin = UM[0,0]*log_g+UM[0,1]*log_b

cv2.imwrite(path + '/Melanin.tif', 255 * Melanin)

#############################################################################################################

Melaninout=Melanin*65535
Melaninout=Melaninout.astype('uint16')
Melaninout=numpy.where(Melaninout < 50000, Melaninout, 0)
p2, p98 = numpy.percentile(Melaninout, (2,98))
Melaninout = exposure.rescale_intensity(Melaninout, in_range=(p2, p98))

cv2.imwrite(path + '/Res1_Melanin.tif', Melaninout)

Mask = output_rgb.astype('double')
Mask_8bit = numpy.uint8(Mask * 255)
Mask_gray = cv2.cvtColor(Mask_8bit, cv2.COLOR_RGB2GRAY)
Mask_gray = cv2.resize(Mask_gray, (Melaninout.shape[1], Melaninout.shape[0]))

imgMaskMel = cv2.bitwise_and(Melaninout, Melaninout, mask=Mask_gray)
cv2.imwrite(path + '/Res1_Melanin_cropped.tif', imgMaskMel)

plt.title('Melanin')
plt.imshow(Melaninout, cmap='gray')
plt.axis('off')
plt.show()

plt.title('Melanin Mask')
plt.imshow(imgMaskMel, cmap='gray')
plt.axis('off')
plt.show()


Blood = UM[1,0]*log_g+UM[1,1]*log_b
#cv2.imwrite(path + '/Blood.tif', 255 * Blood)
Bloodout=Blood*65535
Bloodout=Bloodout.astype('uint16')
Bloodout=numpy.where(Bloodout < 50000, Bloodout, 0)
p2, p98 = numpy.percentile(Bloodout, (2,98))
Bloodout = exposure.rescale_intensity(Bloodout, in_range=(p2, p98))
cv2.imwrite(path + '/Res2_Hemoglobin.tif', Bloodout)

imgMaskHem = cv2.bitwise_and(Bloodout, Bloodout, mask = Mask_gray)
cv2.imwrite(path + '/Res2_Hemoglobin_cropped.tif', imgMaskHem)

plt.title('Hemoglobin')
plt.imshow(Bloodout, cmap='gray')
plt.axis('off')
plt.show()

plt.title('Hemoglobin Mask')
plt.imshow(imgMaskHem, cmap='gray')
plt.axis('off')
plt.show()

#############################################################################################################

# Convert Mask to double
Mask = output_rgb.astype('double')
hsv_mask = cv2.cvtColor(Mask.astype(np.uint8), cv2.COLOR_RGB2HSV)

# Red color range
minRed = numpy.array([ 0, 100, 100], dtype=numpy.uint8)  
maxRed = numpy.array([10, 255, 255], dtype=numpy.uint8)  

mask_lduec = cv2.inRange(hsv_mask, minRed, maxRed)

if numpy.all(mask_lduec == 0):
    Mel_score = 0
    Hem_score = 0
else:
    # Apply mask to Melanin and Blood
    imgMask = cv2.bitwise_and(Mask, Mask, mask=mask_lduec)
    imgMask = imgMask.astype('double')
    Mel_masked = cv2.bitwise_and(Melanin, Melanin, mask=mask_lduec)
    Hem_masked = cv2.bitwise_and(Blood, Blood, mask=mask_lduec)

    # Calculate and print scores
    mask_nonzero = mask_lduec != 0
    Mel_score = numpy.sum(Mel_masked) / numpy.sum(mask_nonzero)
    Hem_score = numpy.sum(Hem_masked) / numpy.sum(mask_nonzero)

MelArray.append(Mel_score)
HemArray.append(Hem_score)
print('                       mask_lduec     ')
print('Melanin                              =', Mel_score)
print('Hemoglobin                           =', Hem_score)

#############################################################################################################

# Blue color range
minBlue = numpy.array([110, 100, 100], dtype=numpy.uint8) 
maxBlue = numpy.array([130, 255, 255], dtype=numpy.uint8)  

mask_rduec = cv2.inRange(hsv_mask, minBlue, maxBlue)

if numpy.all(mask_rduec == 0):
    Mel_score = 0
    Hem_score = 0
else:
    # Apply mask to Melanin and Blood
    imgMask = cv2.bitwise_and(Mask, Mask, mask=mask_rduec)
    imgMask = imgMask.astype('double')
    Mel_masked = cv2.bitwise_and(Melanin, Melanin, mask=mask_rduec)
    Hem_masked = cv2.bitwise_and(Blood,   Blood,   mask=mask_rduec)

    # Calculate and print scores
    mask_nonzero = mask_rduec != 0
    Mel_score = numpy.sum(Mel_masked) / numpy.sum(mask_nonzero)
    Hem_score = numpy.sum(Hem_masked) / numpy.sum(mask_nonzero)

MelArray.append(Mel_score)
HemArray.append(Hem_score)
print('                       mask_rduec    ')
print('Melanin                              =', Mel_score)
print('Hemoglobin                           =', Hem_score)

############################################################################################################

# Green color range
minGreen = numpy.array([50, 100, 100], dtype=numpy.uint8)  
maxGreen = numpy.array([70, 255, 255], dtype=numpy.uint8) 


mask_lcheek = cv2.inRange(hsv_mask, minGreen, maxGreen)

if numpy.all(mask_lcheek  == 0):
    Mel_score = 0
    Hem_score = 0
else:
    # Apply mask to Melanin and Blood
    imgMask = cv2.bitwise_and(Mask, Mask, mask=mask_lcheek)
    imgMask = imgMask.astype('double')
    Mel_masked = cv2.bitwise_and(Melanin, Melanin, mask=mask_lcheek)
    Hem_masked = cv2.bitwise_and(Blood, Blood, mask=mask_lcheek)

    # Calculate and print scores
    mask_nonzero = mask_lcheek != 0
    Mel_score = numpy.sum(Mel_masked) / numpy.sum(mask_nonzero)
    Hem_score = numpy.sum(Hem_masked) / numpy.sum(mask_nonzero)

MelArray.append(Mel_score)
HemArray.append(Hem_score)
print('                       mask_lcheek   ')
print('Melanin                              =', Mel_score)
print('Hemoglobin                           =', Hem_score)

#############################################################################################################




# Define the lower and upper bounds for yellow color in HSV
minYellow= np.array([20, 100, 100], dtype=np.uint8)
maxYellow = np.array([40, 255, 255], dtype=np.uint8)

mask_rcheek = cv2.inRange(hsv_mask , minYellow, maxYellow)


# Check if mask_rcheek is a black image
if numpy.all(mask_rcheek == 0):
    Mel_score = 0
    Hem_score = 0
else:
    # Apply mask to Melanin and Blood
    imgMask = cv2.bitwise_and(Mask, Mask, mask=mask_rcheek)
    imgMask = imgMask.astype('double')
    Mel_masked = cv2.bitwise_and(Melanin, Melanin, mask=mask_rcheek)
    Hem_masked = cv2.bitwise_and(Blood, Blood, mask=mask_rcheek)

    # Calculate and print scores
    mask_nonzero = mask_rcheek != 0
    Mel_score = numpy.sum(Mel_masked) / numpy.sum(mask_nonzero)
    Hem_score = numpy.sum(Hem_masked) / numpy.sum(mask_nonzero)

MelArray.append(Mel_score)
HemArray.append(Hem_score)
print('                       mask_rcheek ')
print('Melanin                              =', Mel_score)
print('Hemoglobin                           =', Hem_score)

#############################################################################################################

# Statistics
print('\n                       Statistics')
file.write('\n\nStatistics')

# Mean
print('Melanin Mean                         = ', statistics.mean(MelArray))
print('Hemoglobin Mean                      = ', statistics.mean(HemArray))
file.write('\nMelanin Mean                  = ' + str(statistics.mean(MelArray)))
file.write('\nHemoglobin Mean               = ' + str(statistics.mean(HemArray)))

# Standard Deviation
print('Melanin Standard Deviation           = ', statistics.stdev(MelArray)) 
print('Hemoglobin Standard Deviation        = ', statistics.stdev(HemArray))
file.write('\nMelanin Standard Deviation    = ' + str(statistics.stdev(MelArray)))
file.write('\nHemoglobin Standard Deviation = ' + str(statistics.stdev(HemArray)))