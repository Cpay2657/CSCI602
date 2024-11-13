from tensorflow.keras.models import Sequential, load_model, Model

model = load_model(r"C:\\Users\\Cpayn\\Desktop\\VSU\\Grad School\\CSCI 602 - Advanced Artificial Intelligence\\Projects\\Project 2\\image_ae\\models\\ae_augmentors")

print(model.history['loss'])