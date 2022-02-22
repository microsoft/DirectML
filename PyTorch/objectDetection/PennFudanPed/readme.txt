1. Directory structure:

PNGImages:   All the database images in PNG format.

PedMasks :   Mask for each image, also in PNG format. Pixels are labeled 0 for background, or > 0 corresponding
to a particular pedestrian ID.

Annotation:  Annotation information for each image.  Each file is in the following format (take FudanPed00001.txt as an example):

# Compatible with PASCAL Annotation Version 1.00
Image filename : "PennFudanPed/PNGImages/FudanPed00001.png"
Image size (X x Y x C) : 559 x 536 x 3
Database : "The Penn-Fudan-Pedestrian Database"
Objects with ground truth : 2 { "PASpersonWalking" "PASpersonWalking" }
# Note there may be some objects not included in the ground truth list for they are severe-occluded
# or have very small size.
# Top left pixel co-ordinates : (1, 1)
# Details for pedestrian 1 ("PASpersonWalking")
Original label for object 1 "PASpersonWalking" : "PennFudanPed"
Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)
Pixel mask for object 1 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"

# Details for pedestrian 2 ("PASpersonWalking")
Original label for object 2 "PASpersonWalking" : "PennFudanPed"
Bounding box for object 2 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (420, 171) - (535, 486)
Pixel mask for object 2 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"

2. Notice
   In [1], we did not label very small, highly occluded pedestrians. 
However in this release of the dataset, we have labeled these pedestrians for future detection.
We list the newly-labeled pedestrians in the file "added-object-list.txt".

   Please download PASCAL toolkit (http://www.pascal-network.org/challenges/VOC/PAScode.tar.gz) to view the annotated pedestrians.
   
3. Acknowledgement

This material is presented to ensure timely dissemination of scholarly and technical work.
Copyright and all rights therein are retained by authors or by other copyright holders. 
 All persons copying this information are expected to adhere to the terms and constraints invoked by 
 each author's copyright. In most cases, these works may not be reposted without 
 the explicit permission of the copyright holder.

4. Related publication
[1] Object Detection Combining Recognition and Segmentation. Liming Wang, Jianbo Shi, Gang Song, I-fan Shen. To Appear in Eighth Asian Conference on Computer Vision(ACCV) 2007
