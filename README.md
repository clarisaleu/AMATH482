# AMATH 482 - Computational Methods For Data Analysis
This repository contains all of the projects for AMATH482 taught by Jason Bramburger (jbrambur@uw.edu) at the University of Washington - Winter 2021.

### Assignment Descriptions:
- **Assignment #1 - A Submarine Problem** <br/>
*In this paper, we explore a dataset containing acoustic data from the Puget Sound taken over a 24-hour time period in half-hour increments. Our goal is to determine the location and path of a new submarine which emits an unknown acoustic frequency to send our P-8 Poseidon subtracking aircraft to follow the submarine. This is done through the use of the discrete Fourier transform to convert the given data to the frequency domain, averaging the transformed frequency data to determine the frequency signature of the submarine, filtering the data with a Gaussian function around the frequency signature to amplify the submarines signal and diminish any noise, and mapping the submarines path and location with the transformed, denoised data for the aircraft to the follow.*<br/>Note, dataset not added to repository due to size.

- **Assignment #2 - Rock & Roll and the Gabor Transform** <br/>
*In this paper, we explore a portion of two of the greatest rock and roll songs of all time, Sweet Child O’ Mine by the Guns N’ Roses and Comfortably Numb by Pink Floyd, in order to analyze the music signals and identify the notes being played in each song from different instruments. Our aim in this exploration is to demonstrate the usefulness of the Gabor Transform in extracting frequency information from time-varying signals, while preserving information about where those frequencies occur in time. We explore the ideas of oversampling and under-sampling signals, the Fourier Transform, the Gabor Transform, and window functions in this paper - as well as provide directions for future work on this problem.*

- **Assignment #3 - PCA and a Spring-Mass System** <br/>
*In this paper, we explore a dataset containing video data from four different oscillation scenarios of a spring-mass system. Each scenario was filmed from three cameras, each in different locations/from different angles. Our goal is to illustrate various aspects of Principal Component Analysis (PCA), its practical usefulness, and the effects of noise on PCA algorithms through performing the PCA method on the given dataset. We explore the ideas of video/image filtering and principal component analysis in this paper - as well as provide directions for future work on this problem and advantages/disadvantages of PCA.<br/>* Note, dataset not added to repository due to size.

- **Assignment #4 - Classifying Digits** <br/>
*In this paper, we explore the MNIST database of handwritten digits, which contains a training set of 60,000 handwritten digits
and a test set of 10,000 handwritten digits. Our goal is to first use principal component analysis (PCA) to perform a low-rank
approximation on each image in the training dataset, which will tell us the most important features in the images, and transform
the dataset into PCA space. After transforming our training and test data into PCA space in order to improve computational
performance - we perform the supervised classification method Linear Discriminant Analysis (LDA) on each pair of digits and
each triplet of digits in order to determine the easiest & most difficult pair and triplet of digits to separate when using LDA.
Additionally - we also perform two supervised machine learning algorithms on the data in PCA space - Support Vector Machine
(SVM) and Decision Tree Learning (DTL) in order to compare and contrast between the three methods of classification between
two and three digits. We also look at how well LDA, SVM, and DTL perform when trying to separate all ten digits. We explore the
ideas of principal component analysis, linear discriminant analysis, support vector machines, and decision tree classifiers in this
paper - as well as provide directions for future work on this problem.*<br/>Note, dataset can be found at http://yann.lecun.com/exdb/mnist/.

- **Assignment #5 - Background Subtraction in Video Streams** <br/>
*In this paper, we look at two different videos - where the first video is an eight-second long clip of a skier dropping
down a mountain and the second video is a six-second long clip of the Monte Carlo race for Formula 1. Our goal
when analyzing these two videos is to apply the concepts of dynamic mode decomposition (DMD) in order to separate
the background of these videos from the foreground - where DMD is a very powerful tool/algorithm when it comes to
the analysis of nonlinear systems. We explore the ideas of video processing, singular value decomposition (SVD), and
DMD in this paper - and provide directions for future work on this problem as well.*
