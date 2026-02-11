Brain Hemorrhage Detector (Personal Project)
About this Project
This is a personal project I built to learn about Artificial Intelligence and how it can help in medicine. The goal was to create a simple tool that can look at a brain CT scan and tell if there is a hemorrhage (bleeding) or if it is normal.

I used Deep Learning (CNN) to make this work. It was a great way to understand how computers can "see" medical images.

Why I Built This
Brain hemorrhages are dangerous, and I wanted to see if I could build a basic AI model that could detect them. It is not a replacement for a doctor, but it shows how powerful Python and AI can be for solving real-world problems.

Tools I Used
Python: For writing the code.

TensorFlow/Keras: To build and train the brain of the AI.

Streamlit: To make the website where you can upload images.

VS Code: The editor I used to write everything.

How to Run It
If you want to try this on your own computer:

Get the code: Download this folder to your PC.

Install the requirements: Open your terminal in this folder and run:

pip install tensorflow streamlit pillow numpy
Start the App: Run this command:

streamlit run app.py
Test it: The website will open. Upload a brain scan image (I included some in the test_images folder if you need them), and watch the AI predict the result!

Dataset
I used a dataset from online sources (Kaggle) which had about 5,000 images of brain scans.

Hemorrhage: Scans with bleeding.

Normal: Healthy scans.

