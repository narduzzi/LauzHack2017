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
1) Align all images with a reference : ECC
2) Default recognition by ROI : text areas
3) Threshold, Normalize

Other pists explored :
SIFT, ORD...


<h2>Results</h2>
Mainly letters default are detected. Some false negative appear, due to differences between images.