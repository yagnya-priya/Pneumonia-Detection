# Pneumonia-Detection
In this project, Python was utilized to develop a web application that aids in the diagnosis of pneumonia using deep learning models. The application, built using the Flask framework, integrates machine learning, database management, and image processing techniques to provide a comprehensive tool for medical diagnosis.

The project begins by loading a pre-trained deep learning model (VGG16) using TensorFlow. This model, stored in the `model.h5` file, is specifically trained to classify chest X-ray images as either showing pneumonia or being normal. Loading the model with TensorFlow ensures that the application is ready to perform predictions when the web server is launched.

To manage patient data, a SQLite database was created and integrated into the application. The database (`patients.db`) is initialized using a custom function, `init_db()`, which sets up a table to store patient details such as name, age, gender, phone number, diagnosis result, probability score, and medical advice. This ensures that all data is persistently stored and can be retrieved or reviewed later.

The core functionality of the application is encapsulated in the `/evaluate` route, where the user can upload a chest X-ray image. The image is processed using OpenCV to convert it into a format suitable for the model. Specifically, the image is converted to grayscale, resized, and normalized before being fed into the model. The model then predicts the likelihood of pneumonia, and based on the prediction, the application provides a diagnosis along with advice.

Moreover, the application also stores the results in the SQLite database, allowing for easy tracking of patient diagnoses over time. This functionality is exposed in the `/patients` route, which retrieves and displays all the patient records stored in the database, offering a clear overview of the diagnostic history.

Finally, the application is made accessible through a web interface, allowing users to interact with the model, submit images for evaluation, and view patient records. This project exemplifies the power of Python in integrating machine learning, database management, and web development to create a practical, real-world application that can assist in medical diagnostics.
