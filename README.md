<p align="center">
<img src="SZCHUAN.png" width="400" text-align="center">
</p>

# LauzHack 2017 - Szechuan Source
Repository for LauzHack 2017 (by Szechuan Source)
Bobst challenge

<h2>Contexte</h2>
<p>The goal is to detect default in packaging. 
 These packaging are provided in image format (5000 images of 25 packages).<p>

<h2>Problematic</h2>
<p>The images are not aligned, and the defaut are multiple.
Defaults :  Letter missing, impression ink stain</p>

<h2>Methodology</h2>
1) Align all images with a reference using ECC algorithms

2) ROI Parsing : Apply ECC by ROI (text areas) 

3) Normalize, Threshold, Substract to get the difference

4) Save the BW image with "activated" areas that correspond to defects.

5) Crop ROI to get original missing/added/swapped letter

Other pists explored :
SIFT, ORB, Standard Deviation...


<h2>Results</h2>
Mainly deleted letters are detected. Some false positive appear (dots, spots on the image). 
We also managed to detect some swapped letters.
