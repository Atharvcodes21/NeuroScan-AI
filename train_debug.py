import tensorflow as tf
# Create shortcuts so you don't have to rewrite your whole code
layers = tf.keras.layers
models = tf.keras.models
import os

# --- SAFETY CHECK 1: VERIFY FOLDERS ---
DATASET_PATH = 'dataset'
print(f"📂 Looking for data in: {os.path.abspath(DATASET_PATH)}")
if not os.path.exists(DATASET_PATH):
    print("❌ ERROR: 'dataset' folder not found!")
    exit()

# --- LOAD DATA ---
BATCH_SIZE = 16 # Smaller batch size helps small datasets learn better
IMG_SIZE = (128, 128)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# --- SAFETY CHECK 2: PRINT THE CLASS ORDER ---
class_names = train_ds.class_names
print("\n" + "="*40)
print(f"🧠 CRITICAL: The AI thinks...")
print(f"   Class 0 = {class_names[0]}")
print(f"   Class 1 = {class_names[1]}")
print("="*40 + "\n")
# COPY THIS ORDER INTO YOUR APP.PY LATER!

# --- BUILD A ROBUST MODEL ---
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    # Layer 1
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Layer 2
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    # Layer 3
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # SAFETY CHECK 3: Dropout prevents memorization
    layers.Dense(len(class_names))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# --- TRAIN ---
print("🚀 Training starting... Watch the 'accuracy' number!")
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=15
)

model.save('brain_hemorrhage_model.h5')
print("\n✅ Saved new model.")