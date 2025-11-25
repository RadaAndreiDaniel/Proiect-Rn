import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
import tempfile

# -----------------------------
# 1️⃣ Folder-ele de antrenament
# -----------------------------
train_dir = r"C:\Users\DANI\OneDrive\Desktop\ProiectRn\dataset\train"
classes = ['mixt', 'iarna', 'vara']  # aceeasi ordine

# -----------------------------
# 2️⃣ Data augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# 3️⃣ Creare model transfer learning
# -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Blocare stratelor pre-antrenate
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 4️⃣ Antrenare model
# -----------------------------
model_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectRn\model\anvelope_model.h5"
if os.path.exists(model_path):
    print("Model incarcat din fisierul salvat.")
    model = load_model(model_path)
else:
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save(model_path)
    print("Model salvat in:", model_path)

# -----------------------------
# 5️⃣ Testare poza noua (convertire automata in JPG)
# -----------------------------
img_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectRn\dataset\test\test4.jpg"  # schimba cu poza ta

# Incarcare imagine si convertire in JPG
img = Image.open(img_path).convert('RGB')
tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
img.save(tmp_file.name, format='JPEG')

# Pregatire imagine
img = Image.open(tmp_file.name)
img = img.resize((224,224))
img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

# Prezicere
pred = model.predict(img_array)
predicted_class = classes[np.argmax(pred[0])]

# Afisare
plt.imshow(np.array(img))
plt.axis('off')
plt.title(f"Clasa prezisa: {predicted_class}")
plt.show()
print("Clasa prezisa pentru poza noua:", predicted_class)


print("Indexi clase:", train_gen.class_indices)
