# MovieAI: Movie Genre Classifier
Teamid: 30 , Project ID = 30
![MovieAI](static/B2.gif)

**MovieAI** is a web-based application that predicts movie genres from poster images using a deep learning model. Powered by PyTorch and Flask, it features a sleek, Netflix-inspired UI with drag-and-drop image uploads, real-time predictions, and a history of past predictions. The model uses a ResNet-18 backbone for multi-label genre classification, trained on a dataset of movie posters and genres.

## Features
- **Drag-and-Drop Upload**: Upload movie posters via a user-friendly drag-and-drop interface.
- **Real-Time Predictions**: Predict multiple genres with confidence scores using a pre-trained ResNet-18 model.
- **Prediction History**: View past predictions with poster thumbnails, genres, confidence scores, and timestamps.
- **Netflix-Inspired UI**: Modern design with Poppins and Montserrat fonts, a dark theme, and responsive layout.
- **Robust Backend**: Flask handles image uploads and predictions, with error handling and logging.
- **Extensible Scripts**: Includes scripts for training (`train.py`), prediction (`predict.py`), dataset utilities (`utils.py`), and model definition (`model.py`).

## Requirements
- Python 3.8+
- PyTorch (`torch`, `torchvision`)
- Flask
- Pillow (`PIL`)
- Pandas
- NumPy
- Scikit-learn
- Requests
- A dataset of movie posters and a CSV file (`MovieGenre.csv`) with columns: `imdbId`, `Poster` (optional URL), `Genre` (pipe-separated genres, e.g., "Action|Adventure").

## Installation
1. **Clone the Repository**:
   ```bash

##Execution
To train the genre prediction: run 
    python train.py
To generate predictions in terminal run python predict.py data/filename.jpg

To verify the accuracy of the model and verify the genre of the poster use the csv file which has the same IMDBid = filename.jpg
The csv file maps the poster to the id

TO run the application run 
    python app.py 
    It runs through flask, the api depicts the history of posters predicted