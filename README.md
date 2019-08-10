# Face-recognition

Developed a face recognition algorithm which identifies the face of a person and check if its a fake or real image of a person. 
Face recognition is done by first creating a dataset of images and passing it through a pre-trained embedding model which creates a 128 point vector representation of the whole image. 
A new face is then compared to this dataset after encoding it, and the face whose embeddings distance is minimum is the recognized face. 
To identify spoof or real image, a separate classification model is built using CNN with over 200 training images of Optum employees.
 
