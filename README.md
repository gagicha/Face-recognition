I am using face-recognition library which provides with a python interface and at the back uses dlib, openFace, OpenCV.

Steps:

Face Detection: detects faces in the image given.
Face Encoding: convert that face image into numerical feature points that describe the face in the best possible way.
Comparison: Compare this feature vector to other faces vectors stored in the database. One that is closest to the input image vector and whose distance is less than the threshold is the predicted output.
Spoofing: Check if the face is real or a spoofed image.

Step1: Face Detection

dlib’s HOG(histogram of oriented gradient) detector is used for this.
Hog detector checks for each pixel in the image the surrounding pixels to the current pixel. It then draws arrows in the direction in which the image is getting darker. In the end, the entire image is filled with arrows. These arrows are called gradients and they show the flow from light to dark across the image. W echeck this pattern to the earlier known face pattern of gradients, and extract only the face part.
I have used the HOG face detection module, which is the default for face-recognition library’s get face location function.

Step2: Encoding Face

Now instead of comparing the entire face to al the faces in the database, we only compare some extracted features from the unknown face to all the known faces. For example, we might measure the size of each ear, the spacing between the eyes, the length of the nose, etc.
The most reliable way to measure a face(Triplet loss)
Ok, so which measurements should we collect from each face to build our known face database? Ear size? Nose length? Eye color? Something else?
It turns out that the measurements that seem obvious to us humans (like eye color) don’t really make sense to a computer looking at individual pixels in an image. Researchers have discovered that the most accurate approach is to let the computer figure out the measurements to collect itself. Deep learning does a better job than humans at figuring out which parts of a face are important to measure. We will train a deep neural network to generate 128 measurements from each face.
The training process works by looking at 3 face images at a time:
Load a training face image of a known person
Load another picture of the same known person
Load a picture of a totally different person
Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2 are slightly closer while making sure the measurements for #2 and #3 are slightly further apart. After repeating this step millions of times for millions of images of thousands of different people, the neural network learns to reliably generate 128 measurements for each person. Any ten different pictures of the same person should give roughly the same measurements.
I am using OpenFace’s library ResNet based pre-trained neural network to get 128 measurements from each face.

Step3: Comparison

We check our face-encoding to the encodings stored in the database. I have used Euclidean distance to check which is the closest match.

Step4: Checking spoof

The image can be of a person or an image of the person’s image, i.e it can be a spoof. So to increase security I added a spoofing layer.
I used in-house employees data to recognize if the image is real or fake. A 4 layer CNN network was trained using Adam optimizer minimizing cross-entropy loss. I got a testing accuracy of 80% with just 50 training examples.
I have used facial recognition using one-shot learning by a deep neural network.
What is one shot learning?
In one shot learning, only one image per person is stored in the database, which is passed through the neural network to generate an embedding vector. This embedding vector is compared with the vector generated for the person who has to be recognized. If there exist similarities between the two vectors then the system recognizes that person, else that person is not there in the database. 
