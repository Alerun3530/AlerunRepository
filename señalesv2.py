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

# Crear el modelo Sequential
modelo_sequential = tf.keras.Sequential()

# Añadir una capa Input explícita para definir la forma de la entrada
modelo_sequential.add(tf.keras.layers.InputLayer(input_shape=(300, 300, 3)))

# Cargar MobileNetV2 sin la parte superior (include_top=False) y con pesos preentrenados
mobilenetv2_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# Añadir el modelo MobileNetV2 al Sequential
modelo_sequential.add(mobilenetv2_base)

# Descongelar algunas capas (últimas 30) para ajuste fino
for layer in mobilenetv2_base.layers[-30:]:
    layer.trainable = True

# Añadir la capa GlobalAveragePooling2D
modelo_sequential.add(tf.keras.layers.GlobalAveragePooling2D())

# Añadir una capa densa con 128 unidades y activación 'relu'
modelo_sequential.add(tf.keras.layers.Dense(128, activation='relu'))

# Añadir Dropout para regularización
modelo_sequential.add(tf.keras.layers.Dropout(0.5))

# Añadir la capa de salida con 18 unidades y activación 'softmax'
modelo_sequential.add(tf.keras.layers.Dense(17, activation='softmax'))

# Definir la tasa de aprendizaje usando una programación exponencial
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)

# Compilar el modelo con optimizador Adam y función de pérdida categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
modelo_sequential.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
modelo_sequential.summary()




# %% 
# Entrenamiento
epocas = 40 

historial = modelo_sequential.fit( 
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
    prediccion = modelo_sequential.predict(img.reshape(-1, 300, 300, 3)) 

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
    prediccion = modelo_sequential.predict(img.reshape(-1, 300, 300, 3)) 

    # Devolver la clase con la mayor probabilidad
    return np.argmax(prediccion[0], axis=-1)


# %% 
# Obtener etiquetas de las clases
class_indices = dataGenEntrenamiento.class_indices 
class_labels = list(class_indices.keys()) 
print("Etiquetas de las clases:", class_labels) 

# %% 
# 0 = no parqueo, 1 = pare , 2 = zona de peatones
url = "https://tienda.semex.com.mx/cdn/shop/products/SenaldepasopeatonalSP-32_700x700.jpg?v=1608229045" 
prediccion = categorizar(url) 



# %%

clases = {
    0: '0101-Parada_De_Bus',
    1: '0201-No_Pase',
    2: '0202-No_Parqueo',
    3: '0203-Pare',
    4: '0204-No_Girar_U',
    5: '0205-No_Parqueo_Deteccion_Electronica',
    6: '0206-Ceda_El_Paso',
    7: '0207-Prohibido_Girar_Derecha',
    8: '0208-Prohibido_Girar_Izquierda',
    9: '0211-Prohibido_Dejar_Pasajeros',
    10: '0212-Velocidad_Maxima',
    11: '0213-Maltrato_Animal',
    12: '0301-Arroyo',
    13: '0302-Tráfico_Bicicletas',
    14: '0303-Zona_De_Peatones',
    15: '0304-Reductor_De_Velocidad',
    16: '0305-Zona_Escolar'
}
nombre_clase = clases[prediccion]

print(f"La predicción es: {nombre_clase}")
# %%
modelo_sequential.save('Reconocimiento_Señales_TransitoV2.h5')
# %%



# %%
