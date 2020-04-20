# Image_prosessing
Image prosessing for IR pictures

Add to project and import

```python
from image_prosessing import *
```

## image_operations.py
Functions to use opnCV to read pictures

Read image from path:
```python
read_image_from_path(path, name, ext, amount)
```
Returns image.
Read image to greyscale:
```python
read_image(img):
```
Returns image in grayscale
Function to invert pictures using bitwise or:
```python
invert_image(img):
```
Returns inverted image.
## filters.py
Functions to greyscale pictures and threshold functions (binary and otzu)

To check if an image is 2D:
```python
check_2D(img)
```
Should return '''True'''/'''False''', not implemented.
Manually created otzu_filter (just works better then openCV filter):
```python
manual_otsu_binary(img):
```
returns filtered image.
OpenCV implemented otzu filter:
```python
otsu_binary(img):
```
returns filtered image.
Filter for morphology operations:
```python
morphology_filter(img, kernelSize):
```
Function to tie the above together and let user decide what filter to use:
```python
filter_img(img, filterType=0, morphology=False):
```
Also has example function to test:
```python
filters_test_func():
```
## blob_detection.py
Functions to use simple blob detector from OpenCV

Function to create simple blob detector from parameters set globaly:
```python
init_blob_detector():
```
returns blob detector.
Function to draw detected keypoints on original image after blob detection:
```python
draw_blobs(img, keyPoints): 
```
Blob detection function:
```python
blob_detection(img):
```
returns list of keypoints -> list of blobs in image.

Also has example function - has not been implemented yet:
```python
blob_detection_test_func():
```
