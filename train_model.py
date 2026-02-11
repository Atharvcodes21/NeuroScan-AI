import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
import os

# --- 1. CONFIGURATION ---
DATASET_PATH = 'dataset'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10 

print("🚀 Initializing Safe-Mode MobileNetV2...")

# --- 2. DATA LOADING ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- 3. LOAD GOOGLE'S PRE-TRAINED BRAIN ---
base_model = applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base to keep training fast on your CPU
base_model.trainable = False 

# --- 4. BUILD THE MODEL (SAFE MODE) ---
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    
    # ✅ THE FIX: Standard layer. 
    # This scales pixels from [0, 255] to [-1, 1] exactly like MobileNet needs.
    # It saves perfectly inside the .h5 file.
    layers.Rescaling(1./127.5, offset=-1), 

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Prevents memorization
    layers.Dense(2) # Output: [bleed, normal]
])

# --- 5. COMPILE & TRAIN ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print("\n🔥 Training started. Please wait for 10 epochs...")
model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS
)

# --- 6. SAVE ---
# Delete old file just in case
if os.path.exists('brain_hemorrhage_model.h5'):
    os.remove('brain_hemorrhage_model.h5')

model.save('brain_hemorrhage_model.h5')
print("\n✅ DONE! New 'brain_hemorrhage_model.h5' saved.")
print("👉 You can now run 'streamlit run app.py' without errors.")