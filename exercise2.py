#https://developers.google.com/codelabs/tensorflow-2-computervision#1

#2. Começar a programar
import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

#separando os dados
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

print(training_images[0])
print(training_labels[0])

#normalizando
training_images  = training_images / 255.0
test_images = test_images / 255.0

#3. Projetar o modelo
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#4. Compile e treine o modelo
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
#1875/1875 [==============================] - 13s 7ms/step - loss: 0.2949 - accuracy: 0.8915

#5. Testar o modelo
model.evaluate(test_images, test_labels)
#313/313 [==============================] - 2s 4ms/step - loss: 0.3485 - accuracy: 0.8761

#6. Exercícios de exploração

classifications = model.predict(test_images)
print(classifications[0])