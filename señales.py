# %%
import os 
import matplotlib.pyplot as plt 
import matplotlib.image as mping 

plt.figure(figsize=(15,15)) 

carpeta = "C:/Users/aleco/Downloads/SeñalesDeTransito/dataset/0203-Pare" 
imagenes = os.listdir(carpeta) 

for i , nombreimg in enumerate(imagenes[:10]): 
    plt.subplot(1,10,i+1) 
    img = mping.imread(carpeta + "/" + nombreimg) 
    plt.imshow(img)

# %% 
# Aumento de datos avanzado
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np 

datagen = ImageDataGenerator( 
    rescale = 1. / 255, 
    rotation_range = 30, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 15, 
    zoom_range = [0.5, 1.2], 
    brightness_range=[0.8, 1.2], 
    validation_split = 0.2 
) 

dataGenEntrenamiento = datagen.flow_from_directory( 
    'C:/Users/aleco/Downloads/SeñalesDeTransito/dataset', 
    target_size = (300,300), 
    batch_size = 32, shuffle = True, subset = "training") 

dataGenValidacion = datagen.flow_from_directory( 
    'C:/Users/aleco/Downloads/SeñalesDeTransito/dataset', 
    target_size = (300,300), 
    batch_size = 32, shuffle = True, subset = "validation") 

# %% 
for imagen, etiqueta in dataGenEntrenamiento: 
    for i in range(5): 
        plt.subplot(1,5,i+1) 
        plt.xticks([]) 
        plt.yticks([]) 
        plt.imshow(imagen[i]) 
    break 
plt.show() 

# %% 
import tensorflow as tf 

# Cargar el modelo MobileNetV2 preentrenado de TensorFlow
mobilenetv2_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(1, 300, 300, 3)) 

# Descongelar algunas capas (últimas 30) para ajuste fino
for layer in mobilenetv2_base.layers[-30:]: 
    layer.trainable = True 

# Definir la entrada
inputs = tf.keras.Input(shape=(300, 300, 3)) 

# Pasar la entrada a través del modelo base
x = mobilenetv2_base(inputs, training=True) 

# Añadir capas adicionales
x = tf.keras.layers.GlobalAveragePooling2D()(x) 
x = tf.keras.layers.Dense(128, activation='relu')(x)  # Nueva capa completamente conectada
x = tf.keras.layers.Dropout(0.5)(x)  # Regularización con Dropout

# Añadir la capa final de salida
outputs = tf.keras.layers.Dense(18, activation='softmax')(x) 

# Crear el modelo funcional
modelo = tf.keras.Model(inputs=inputs, outputs=outputs) 

# Compilar el modelo con una tasa de aprendizaje ajustada
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( 
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9) 
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) 
modelo.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 

# %% 
modelo.summary() 

# %% 
# Entrenamiento
epocas = 40 

historial = modelo.fit( 
    dataGenEntrenamiento, epochs=epocas, batch_size=32, 
    validation_data=dataGenValidacion 
) 

# %% 
# Gráficas de precisión y pérdida
acc = historial.history['accuracy'] 
val_acc = historial.history['val_accuracy'] 

loss = historial.history['loss'] 
val_loss = historial.history['val_loss'] 

rango_epocas = range(40) 

plt.figure(figsize=(8,8)) 

plt.subplot(1,2,1) 
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento') 
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas') 
plt.legend(loc='lower right') 
plt.title('Precisión de entrenamiento y pruebas') 

plt.subplot(1,2,2) 
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento') 
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas') 
plt.legend(loc='upper right') 
plt.title('Pérdida de entrenamiento y pruebas') 

plt.show() 

# %% 
import requests 
from io import BytesIO 
from PIL import Image 
import cv2 

def categorizar(url): 
    # Realizar una solicitud HTTP a la URL
    respuesta = requests.get(url) 

    # Abrir la imagen desde el contenido de la respuesta
    img = Image.open(BytesIO(respuesta.content)) 

    # Convertir la imagen a un array de numpy y normalizar
    img = np.array(img).astype(float) / 255 

    # Redimensionar la imagen
    img = cv2.resize(img, (300, 300)) 

    # Realizar la predicción
    prediccion = modelo.predict(img.reshape(-1, 300, 300, 3)) 

    # Devolver la clase con la mayor probabilidad
    return np.argmax(prediccion[0], axis=-1) 

# %%
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Asumiendo que ya tienes un modelo cargado
# modelo = load_model('ruta_al_modelo.h5')

def categorizarLocal(ruta_imagen): 
    # Abrir la imagen desde la ruta local
    img = Image.open(ruta_imagen) 

    # Convertir la imagen a un array de numpy y normalizar
    img = np.array(img).astype(float) / 255 

    # Redimensionar la imagen a 224x224
    img = cv2.resize(img, (300, 300)) 

    # Realizar la predicción
    prediccion = modelo.predict(img.reshape(-1, 300, 300, 3)) 

    # Devolver la clase con la mayor probabilidad
    return np.argmax(prediccion[0], axis=-1)


# %% 
# Obtener etiquetas de las clases
class_indices = dataGenEntrenamiento.class_indices 
class_labels = list(class_indices.keys()) 
print("Etiquetas de las clases:", class_labels) 

# %% 
# 0 = no parqueo, 1 = pare , 2 = zona de peatones
url = "https://jopavisos.com/wp-content/uploads/2021/04/Transito-amarillas-2-14.png" 
prediccion = categorizar(url) 

# %%
url = "C:/Users/aleco/OneDrive/Imágenes/Capturas de pantalla/Captura de pantalla 2024-09-30 231358.png" 
prediccion = categorizarLocal(url) 

# %%

clases = {
    0: '0101-Parada_De_Bus',
    1: '0201-No_Pase',
    2: '0202-No_Parqueo',
    3: '0203-Pare',
    4: '0204-No_Girar_U',
    5: '0205-Prohibido_Parquear',
    6: '0206-Ceda_El_Paso',
    7: '0207-Prohibido_Girar_Derecha',
    8: '0208-Prohibido_Girar_Izquierda',
    9: '0210-Deteccion_Electronica',
    10: '0211-Prohibido_Dejar_Pasajeros',
    11: '0212-Velocidad_Maxima',
    12: '0213-Maltrato_Animal',
    13: '0301-Arroyo',
    14: '0302-Tráfico_Bicicletas',
    15: '0303-Zona_De_Peatones',
    16: '0304-Reductor_De_Velocidad',
    17: '0305-Zona_Escolar'
}
nombre_clase = clases[prediccion]

print(f"La predicción es: {nombre_clase}")
# %%
modelo.save('Reconocimiento_Señales_Transito.h5')
# %%

