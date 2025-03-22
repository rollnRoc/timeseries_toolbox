import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import json
import base64
import io
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        return jsonify({
            'message': f'{filename} successfully uploaded',
            'columns': columns,
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview/<filename>', methods=['GET'])
def preview_data(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        df = pd.read_csv(filepath)
        return jsonify({
            'head': df.head(5).to_dict('records'),
            'info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_data(df, target_column, sequence_length):
    df[target_column] = df[target_column].fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[target_column]])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    return X[:train_size], X[train_size:], y[:train_size], y[train_size:], scaler

def build_cnn_model(input_shape, num_filters, kernel_size, dense_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_model(input_shape, lstm_units, dense_units):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=lstm_units, return_sequences=False, input_shape=input_shape),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, scaler, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    predictions = model.predict(X_test)

    y_true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    rmse = sqrt(mse)
    
    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_rescaled, label="Actual Values")
    plt.plot(y_pred_rescaled, label="Predictions", linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Model Predictions")
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'plot': plot_data,
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    try:
        # Load the dataset
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(data['filename']))
        df = pd.read_csv(filepath)
        
        target_column = data['targetColumn']
        model_type = data['modelType']
        params = data['params']
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            df, 
            target_column, 
            params['sequenceLength']
        )
        
        # Reshape data according to model type
        if model_type == 'cnn':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model = build_cnn_model(
                (params['sequenceLength'], 1),
                params['numFilters'],
                params['kernelSize'],
                params['denseUnits']
            )
        elif model_type == 'lstm':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model = build_lstm_model(
                (params['sequenceLength'], 1),
                params['lstmUnits'],
                params['denseUnits']
            )
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Train and evaluate model
        results = train_and_evaluate_model(
            model, 
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            scaler,
            params['epochs']
        )
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)