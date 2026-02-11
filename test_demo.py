import tensorflow as tf
import numpy as np
import sys
import os

# 1. LOAD THE TRAINED BRAIN
model_path = 'brain_hemorrhage_model.h5'

if not os.path.exists(model_path):
    print("❌ Error: I can't find the model file! Did you rename it?")
    sys.exit()

print("🧠 Loading the AI model... (This takes 2 seconds)")
model = tf.keras.models.load_model(model_path)

# 2. DEFINE THE LABELS (Must be alphabetical!)
class_names = ['Hemorrhagic', 'NORMAL'] 

def predict_image(img_path):
    print(f"🔍 Scanning image: {img_path}")
    
    try:
        # Load and resize the image to match the training size
        img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    except:
        print("❌ Error: Could not open image. Check the file path!")
        return

    # Convert image to numbers
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

    # Ask the AI
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Print Result
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print("\n" + "="*30)
    print(f"📢 DIAGNOSIS: {result}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    print("="*30 + "\n")

# Run it
if __name__ == "__main__":
    # If you run via command line: python test_demo.py image.png
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # HARDCODED TEST (Change this filename to test different images!)
        target_image = "scan/test.jpg"
        
    predict_image(target_image)