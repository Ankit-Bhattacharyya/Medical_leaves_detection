import tensorflow as tf
import numpy as np
import gradio as gr

# With the path to your actual .h5 file
model_path = 'model__xception.h5'

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

#Defining the labels
labels = ['Aloevera','Amla','Amruthaballi','Arali','Astma_weed','Badipala','Balloon_Vine','Bamboo','Beans','Betel','Bhrami','Bringaraja','Caricature','Castor','Catharanthus',
          'Chakte','Chilly','Citron lime (herelikai)','Coffee','Common rue(naagdalli)','Coriender','Curry','Doddpathre','Drumstick','Ekka','Eucalyptus','Ganigale','Ganike',
          'Gasagase','Ginger','Globe Amarnath','Guava','Henna','Hibiscus','Honge','Insulin','Jackfruit','Jasmine','Kambajala','Kasambruga','Kohlrabi','Lantana','Lemon',
          'Lemongrass','Malabar_Nut','Malabar_Spinach','Mango','Marigold','Mint','Neem','Nelavembu','Nerale','Nooni','Onion','Padri','Palak(Spinach)','Papaya','Parijatha',
          'Pea','Pepper','Pomoegranate','Pumpkin','Raddish','Rose','Sampige','Sapota','Seethaashoka','Seethapala','Spinach1','Tamarind','Taro','Tecoma','Thumbe','Tomato',
          'Tulsi','Turmeric','ashoka','camphor','kamakasturi','kepala']

## Define the predict function for Gradio
def predict_gradio(image):
    # Preprocess the input image
    img_array = tf.image.resize(image, (299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make a prediction using the trained model
    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.sigmoid(predictions[0])

    return ("This image most likely belongs to {} with a {:.2f} percent confidence.".format(labels[np.argmax(score)], 100 * np.max(score)))

#Create a Gradio interface
iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(),
    outputs="label",
    live=True,
    cache_examples=False
)

# Launch the Gradio interface
iface.launch(share=True)