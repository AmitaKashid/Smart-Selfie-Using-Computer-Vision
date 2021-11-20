# Smart-Selfie-Using-Computer-Vision

## :sunglasses:  About The Project: :point_down:

# Facial Expression Recognition Classifier Model :

Facial expression for emotion detection has always been an easy task for humans, but achieving the same task with a computer algorithm is quite challenging. With the recent advancement in computer vision and machine learning, it is possible to detect emotions from images.In this project,we propose a novel technique called facial emotion recognition using convolutional neural networks,python and flask. Facial expressions are the vital identifiers for human feelings, because it corresponds to the emotions. Most of the times (roughly in 55% cases), the facial expression is a nonverbal way of emotional expression, and it can be considered as concrete evidence to uncover whether an individual is speaking the truth or not.


- **Real-time Video input** <br>
<img width="891" alt="nuetral" src="https://user-images.githubusercontent.com/57671048/98197451-c8192a00-1f4c-11eb-8b9f-e752acce1127.png"><br>

- It predicts the **Emotion of users** and also gives **Graphical Visualization** of **Emotions** as shown above.

## :loop: Tech Stack used :point_down:
- Python
- Flask
- HTML, CSS
- Deep Learning (CNN)

## :boom: Getting Started: Steps to run the Project in your local device !!
- Fork this repository.
- Clone the repository to your System using `git clone`
- Example : `git clone https://github.com<your-github-username>/Facial-Expression-Recognition-Classifier-Model`
- Create a new [Virtual Environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) with python 3.7.0 version. 
- Install all the dependencies with `pip install -r requirements.txt`.
- Now run the `main.py` file. 
- Once it shows `Running on http://127.0.0.1:5000/` go to *http://127.0.0.1:5000/* in your browser.


## :computer: Coding Structure:
1. We will be using a well-known FER dataset. FER dataset contains 35,887 face images, including 28,709 training sets, 3589 verification sets, and 589 test sets, all of which are grayscale images of 48 pixels × 48 pixels. 
2. When the smart selfie application starts, it will start extracting features from frame. 3. Features like eyebrows, eyes, nose get detected using Haar-like features from the frame. 
4. In haar-cascade, each 3x3 kernel moves across the image and does matrix multiplication with every 3x3 part of the image, emphasizing some features and smoothing others. Using sliding windows 160,000+ number of features get extracted. 
5. Series of Haar-cascade classifiers will be used. Each stage of the classifier labels the region defined by the current location of the sliding window as either positive or negative. Positive indicates that an object was found and negative indicates no objects were found. 
6. AdaBoost algorithm will be performed for selecting best features as among all features we calculated as most of them are irrelevant. 
7. Adaboost is an ensemble learning algorithm. Adaboost finds the set of best weak haar-cascade classifiers and combines them to produce a strong classifier that will produce good results on unseen data. In this way it selects the best features from 160,000+ features. By using the selected features, faces will get detected. 
8. Crop the image to the detected face and resize it to 350*350 and save the image. 9. CNN models will be trained on the FER dataset. 
10. On the basis of these features calculated, CNN + VGG-16 algorithm will be used to detect whether the people present in the frame are doing particular expressions like happy, neutral or surprised. 
K.C. College of Engineering & Management Studies & Research, Thane (E) 9
Smart Selfie using Computer Vision 
11. VGG-16 is a pretrained ConvNet that consists of a 16-depth weight layer with very small (3 x 3) convolution filters used for object recognition. It applies convolution operation on each pixel of the images and ultimately generates ’n’ dimensional arrays which are nothing but learnt features of the images known as bottleneck features. 
12. A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. 
13. System will capture images if all people detected are doing preferred expressions. 

<img src="https://miro.medium.com/max/1864/1*oURfHMP1--ttXnDx0heusg.png">

