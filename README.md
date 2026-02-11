Brain Hemorrhage Detection and Triage System
Project Overview
This project presents an automated system for the detection of intracranial hemorrhages in CT scans. The primary goal is to provide a computational tool that can assist in the triage process by analyzing multiple scans and identifying cases that require immediate clinical attention.

Key Features
Automated Triage: The system categorizes scans and generates a priority list based on detected patterns.

Batch Analysis: Supports multiple simultaneous uploads to streamline the diagnostic workflow.

Modern Web Interface: Built for rapid deployment and ease of use in medical settings.

Acknowledgements and Technology Credits
This project leverages state-of-the-art AI architectures and collaborative tools:

MobileNetV2 (Architecture): The core intelligence of this system is powered by the MobileNetV2 Convolutional Neural Network. This project utilizes Transfer Learning on pre-trained weights to adapt the model for specialized medical image classification, ensuring high accuracy with optimized computational efficiency.

Google Gemini (AI Collaboration): The development, debugging, and system architecture were refined through an iterative collaboration with Google Gemini. The AI was instrumental in resolving complex dependency conflicts and optimizing the training and deployment scripts.

TensorFlow and Keras: These frameworks provided the deep learning backbone for training the model on a dataset of 5,000 images.

Streamlit: The user interface and real-time inference dashboard were developed using the Streamlit framework.

Training and Methodology
The system was trained on a comprehensive dataset of 5,000 CT scans. To ensure robust performance, the following methodologies were implemented:

Data Augmentation: Techniques such as random rotation and zooming were applied to improve the model's generalization capabilities.

Normalization: Input data was standardized to a 128x128 resolution for consistent multi-scan analysis.

Setup and Execution
Install requirements:

Bash
pip install tensorflow streamlit pandas pillow
Run the application:

Bash
streamlit run app.py
