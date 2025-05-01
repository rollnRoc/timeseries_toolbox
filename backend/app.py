from flask import Flask, request, jsonify
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import warnings
warnings.filterwarnings('ignore')

# For deep learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# For ARIMA models
from statsmodels.tsa.arima.model import ARIMA

# For Prophet models
from prophet import Prophet

# For XGBoost models
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Mevcut import ifadelerine eklenecek yeni kütüphaneler

# SARIMA için
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Random Forest için
from sklearn.ensemble import RandomForestRegressor

# ROCKET için
# Eklenti kullanmanız gerekirse bu kurulum yapılmalıdır:
# pip install sktime
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

# Shapelet Transform için 
from sktime.transformations.panel.shapelet_transform import ShapeletTransform
from sklearn.ensemble import RandomForestClassifier

# Sınıflandırma metrikleri için
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper functions
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(df, target_column, params):
    # Extract target data
    data = df[target_column].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    seq_length = params.get('sequenceLength', 10)
    X, y = create_sequences(data_scaled, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler

def create_plot(y_true, y_pred, title="Actual vs Predicted"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def create_feature_importance_plot(feature_names, importances):
    plt.figure(figsize=(10, 6))
    
    # Sort importances in descending order
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_importances)), sorted_features)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

# Model training functions
def train_cnn_model(df, target_column, params):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column, params)
    
    # Reshape for CNN [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build CNN model
    model = Sequential([
        Conv1D(filters=params.get('numFilters', 64), 
               kernel_size=params.get('kernelSize', 3), 
               activation='relu', 
               input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(params.get('denseUnits', 64), activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=params.get('epochs', 50),
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv.flatten(), y_pred_inv.flatten(), history

def train_lstm_model(df, target_column, params):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column, params)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build LSTM model
    model = Sequential([
        LSTM(units=params.get('lstmUnits', 50), 
             return_sequences=False, 
             input_shape=(X_train.shape[1], 1)),
        Dense(params.get('denseUnits', 64), activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=params.get('epochs', 50),
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv.flatten(), y_pred_inv.flatten(), history

def train_arima_model(df, target_column, params):
    # Get time series data
    data = df[target_column].values
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # ARIMA parameters
    p = params.get('p', 1)  # AR order
    d = params.get('d', 1)  # Differencing
    q = params.get('q', 1)  # MA order
    
    # Fit ARIMA model
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        
        # Make predictions
        predictions = model_fit.forecast(steps=len(test))
        
        # Get AIC
        aic = model_fit.aic
        
        return test, predictions, aic
    except Exception as e:
        # Fallback to simple model if there's an error
        print(f"ARIMA error: {e}")
        model = ARIMA(train, order=(1, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        aic = model_fit.aic
        
        return test, predictions, aic

def train_prophet_model(df, target_column, params):
    # Prophet requires 'ds' (dates) and 'y' (values) columns
    # Create a copy to avoid modifying original
    prophet_df = df.copy()
    
    # If no date column, create an artificial one
    if 'date' in prophet_df.columns:
        date_col = 'date'
    else:
        # Create a synthetic date index
        prophet_df['date'] = pd.date_range(start='2020-01-01', periods=len(prophet_df))
        date_col = 'date'
    
    # Prepare data for Prophet
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(prophet_df[date_col]),
        'y': pd.to_numeric(prophet_df[target_column], errors='coerce')
    })
    
    # Split data
    train_size = int(len(prophet_data) * 0.8)
    train = prophet_data.iloc[:train_size]
    test = prophet_data.iloc[train_size:]
    
    # Create and fit the model
    model = Prophet(
        seasonality_mode=params.get('seasonalityMode', 'additive'),
        changepoint_prior_scale=params.get('changePointPrior', 0.05),
        seasonality_prior_scale=params.get('seasonalityPrior', 10)
    )
    
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Get predictions for test period
    predictions = forecast.iloc[-len(test):]['yhat'].values
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    y_true = test['y'].values
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    
    return y_true, predictions, mape

def train_xgboost_model(df, target_column, params):
    # 1) Tarihi datetime’a çevir (opsiyonel, Prophet’de kullanmak üzere)
    if 'data' in df.columns:
        df = df.copy()
        df['data'] = pd.to_datetime(df['data'])

    # 2) Lag özelliklerini oluştur
    seq_length = params.get('sequenceLength', 10)
    df_features = df.copy()
    for i in range(1, seq_length + 1):
        df_features[f'lag_{i}'] = df_features[target_column].shift(i)
    df_features = df_features.dropna()

    # 3) Hedef ve özellikleri ayır
    y = df_features[target_column]
    X = df_features.drop(columns=[target_column])

    # 4) Sadece nümerik/bool/category tipte olanları bırak,
    #    data sütununu da bu noktada at
    X = X.select_dtypes(include=['number','bool','category'])

    # 5) Eğitim/test bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 6) Modeli eğit
    params_xgb = {
        'max_depth':       params.get('maxDepth', 6),
        'learning_rate':   params.get('learningRate', 0.1),
        'n_estimators':    params.get('nEstimators', 100),
        'objective':       'reg:squarederror',
        'enable_categorical': True
    }
    start = time.time()
    model = xgb.XGBRegressor(**params_xgb)
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # 7) Tahmin ve değişken önemini döndür
    y_pred = model.predict(X_test)
    fi = model.feature_importances_
    fn = X.columns

    return y_test, y_pred, fi, fn, training_time
def train_transformer_model(df, target_column, params):
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column, params)
    
    # Reshape for Transformer [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Get parameters
    dim_model = params.get('dimModel', 64)
    num_heads = params.get('numHeads', 8)
    num_encoder_layers = params.get('numEncoderLayers', 4)
    dropout_rate = params.get('dropoutRate', 0.1)
    dense_units = params.get('denseUnits', 64)
    
    # Build Transformer model
    inputs = tf.keras.Input(shape=(X_train.shape[1], 1))
    
    # Add positional encoding
    x = inputs
    
    # Encoder layers
    for _ in range(num_encoder_layers):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=dim_model // num_heads
        )(x, x)
        
        # Add & Norm
        attention_output = Dropout(dropout_rate)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ffn = Dense(dim_model * 4, activation='relu')(x)
        ffn = Dense(dim_model)(ffn)
        
        # Add & Norm
        ffn = Dropout(dropout_rate)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = Dense(dense_units, activation='relu')(x)
    outputs = Dense(1)(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=params.get('epochs', 50),
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv.flatten(), y_pred_inv.flatten(), history

# 1. SARIMA Model Fonksiyonu
def train_sarima_model(df, target_column, params):
    # Get time series data
    data = df[target_column].values
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # SARIMA parameters
    p = params.get('p', 1)  # AR order
    d = params.get('d', 1)  # Differencing
    q = params.get('q', 1)  # MA order
    P = params.get('P', 1)  # Seasonal AR order
    D = params.get('D', 0)  # Seasonal differencing
    Q = params.get('Q', 1)  # Seasonal MA order
    s = params.get('s', 12)  # Seasonal period
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Fit SARIMA model
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        
        # Make predictions
        predictions = model_fit.forecast(steps=len(test))
        
        # Get AIC
        aic = model_fit.aic
        
        return test, predictions, aic
    except Exception as e:
        # Fallback to simple model if there's an error
        print(f"SARIMA error: {e}")
        model = SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12))
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=len(test))
        aic = model_fit.aic
        
        return test, predictions, aic

# 2. Random Forest Model Fonksiyonu
def train_random_forest_model(df, target_column, params):
    from sklearn.ensemble import RandomForestRegressor
    
    # 1) Tarihi datetime'a çevir (opsiyonel)
    if 'data' in df.columns:
        df = df.copy()
        df['data'] = pd.to_datetime(df['data'])

    # 2) Lag özelliklerini oluştur
    seq_length = params.get('sequenceLength', 10)
    df_features = df.copy()
    for i in range(1, seq_length + 1):
        df_features[f'lag_{i}'] = df_features[target_column].shift(i)
    df_features = df_features.dropna()

    # 3) Hedef ve özellikleri ayır
    y = df_features[target_column]
    X = df_features.drop(columns=[target_column])

    # 4) Sadece nümerik/bool/category tipte olanları bırak
    X = X.select_dtypes(include=['number','bool','category'])

    # 5) Eğitim/test bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 6) RandomForest modelini eğit
    params_rf = {
        'n_estimators': params.get('nEstimators', 100),
        'max_depth': params.get('maxDepth', 10),
        'min_samples_split': params.get('minSamplesSplit', 2),
        'min_samples_leaf': params.get('minSamplesLeaf', 1)
    }
    
    start = time.time()
    model = RandomForestRegressor(**params_rf, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # 7) Tahmin ve değişken önemini döndür
    y_pred = model.predict(X_test)
    fi = model.feature_importances_
    fn = X.columns

    return y_test, y_pred, fi, fn, training_time

# 3. Rocket Model Fonksiyonu
def train_rocket_model(df, target_column, params):
    # Rocket için gerekli kütüphaneleri import et
    from sktime.transformations.panel.rocket import Rocket
    from sklearn.linear_model import RidgeClassifierCV
    
    # Veriyi hazırla
    # Zaman serisi verilerini sktime formatına dönüştür
    data = df[target_column].values
    
    # Zaman serisini parçalara böl (pencereler oluştur)
    def create_windows(data, window_size, stride=1):
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i+window_size])
        return np.array(windows)
    
    window_size = params.get('windowSize', 30)
    stride = params.get('stride', 1)
    windows = create_windows(data, window_size, stride)
    
    # Pencerelenmiş veriyi eğitim-test olarak ayır
    split = int(len(windows) * 0.8)
    X_train, X_test = windows[:split], windows[split:]
    
    # Etiketleri oluştur (trend tahmini için)
    # Örneğin, her pencerenin ortalaması bir önceki pencereden büyükse 1, değilse 0
    y_train = np.zeros(len(X_train))
    y_test = np.zeros(len(X_test))
    
    for i in range(1, len(X_train)):
        if np.mean(X_train[i]) > np.mean(X_train[i-1]):
            y_train[i] = 1
    
    for i in range(1, len(X_test)):
        if np.mean(X_test[i]) > np.mean(X_test[i-1]):
            y_test[i] = 1
    
    # X_train ve X_test'i 3 boyutlu yap [örnekler, zaman adımları, özellikler]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Rocket transformasyonunu uygula
    rocket = Rocket(num_kernels=params.get('numKernels', 2000), random_state=42)
    start_time = time.time()
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)
    X_test_transform = rocket.transform(X_test)
    
    # Ridge Classifier ile trend tahmini yap
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)
    
    # Tahmin yap
    y_pred = classifier.predict(X_test_transform)
    training_time = time.time() - start_time
    
    # Sonuçları döndür
    # ROCKET sınıflandırma yapar, ancak regresyon için uyarlanabilir
    # Şimdilik sınıflandırma sonuçlarını dönüyoruz
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    return y_test, y_pred, accuracy, training_time

# 4. InceptionTime Model Fonksiyonu
def train_inception_time_model(df, target_column, params):
    # Veriyi hazırla
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, target_column, params)
    
    # Reshape for InceptionTime [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Model parametreleri
    nb_filters = params.get('nbFilters', 32)
    use_residual = params.get('useResidual', True)
    use_bottleneck = params.get('useBottleneck', True)
    depth = params.get('depth', 6)
    kernel_size = params.get('kernelSize', 3)
    
    # InceptionTime modül fonksiyonlarını tanımla
    def inception_module(input_tensor, stride=1, activation='linear', bottleneck_size=32, kernel_size=3, nb_filters=32):
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size, kernel_size=1, 
                                    padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size, kernel_size * 2, kernel_size * 3]
        conv_list = []

        for i in range(len(kernel_size_s)):
            conv = Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                        strides=stride, padding='same', activation=activation, use_bias=False)(input_inception)
            conv_list.append(conv)

        max_pool_1 = MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = LayerNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                        padding='same', use_bias=False)(input_tensor)
        return tf.keras.layers.Add()([shortcut, out_tensor])

    # InceptionTime modeli oluştur
    input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], 1))
    x = input_layer
    input_res = input_layer

    for d in range(depth):
        x = inception_module(x, kernel_size=kernel_size, nb_filters=nb_filters)
        
        if d % 3 == 2 and use_residual:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)
    output_layer = Dense(1)(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=params.get('epochs', 50),
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    training_time = time.time() - start_time
    
    # Inverse transform
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv.flatten(), y_pred_inv.flatten(), history, training_time

# 5. Shapelet Transform Classifier Fonksiyonu
def train_shapelet_transform_model(df, target_column, params):
    # Shapelet için gerekli kütüphaneyi import et
    from sktime.transformations.panel.shapelet_transform import ShapeletTransform
    from sklearn.ensemble import RandomForestClassifier
    
    # Veriyi hazırla - zaman serisi sınıflandırma için
    data = df[target_column].values
    
    # Zaman serisini parçalara böl (pencereler oluştur)
    def create_windows(data, window_size, stride=1):
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i+window_size])
        return np.array(windows)
    
    window_size = params.get('windowSize', 30)
    stride = params.get('stride', 5)
    windows = create_windows(data, window_size, stride)
    
    # Etiketleri oluştur (trend tahmini için)
    # Örneğin, her pencerenin ortalaması bir önceki pencereden büyükse 1, değilse 0
    y = np.zeros(len(windows))
    for i in range(1, len(windows)):
        if np.mean(windows[i]) > np.mean(windows[i-1]):
            y[i] = 1
    
    # Veriyi eğitim-test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(
        windows, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # 3 boyutlu formata dönüştür [örnekler, zaman adımları, özellikler]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # ShapeletTransform parametreleri
    n_shapelets = params.get('nShapelets', 100)
    shapelet_lengths = params.get('shapeletLengths', [0.1, 0.2, 0.3])
    
    # ShapeletTransform boyut belirleme (shapeletLengths'i tam sayıya dönüştür)
    shapelet_len_int = [int(sl * window_size) for sl in shapelet_lengths]
    shapelet_len_int = [sl for sl in shapelet_len_int if sl > 0]  # 0'dan büyük olanları al
    
    # ShapeletTransform ve sınıflandırıcı modeli oluştur
    start_time = time.time()
    st = ShapeletTransform(
        n_shapelets=n_shapelets,
        shapelet_lengths=shapelet_len_int,
        random_state=42
    )
    
    # ShapeletTransform'u eğit ve veriyi dönüştür
    st.fit(X_train, y_train)
    X_train_st = st.transform(X_train)
    X_test_st = st.transform(X_test)
    
    # RandomForest sınıflandırıcı ile shapelet özelliklerine dayalı tahmin yap
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_st, y_train)
    
    # Tahmin ve değerlendirme
    y_pred = clf.predict(X_test_st)
    training_time = time.time() - start_time
    
    # Doğruluk oranını hesapla
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    return y_test, y_pred, accuracy, training_time
# API Routes
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV
        try:
            df = pd.read_csv(filepath)
            columns = df.columns.tolist()
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'columns': columns
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/preview/<filename>', methods=['GET'])
def preview_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        df = pd.read_csv(filepath)
        
        # Basic info
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist()
        }
        
        # Convert head to list of records for JSON serialization
        head_records = df.head().to_dict('records')
        
        return jsonify({
            'info': info,
            'head': head_records
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    filename = data.get('filename')
    target_column = data.get('targetColumn')
    model_type = data.get('modelType', 'cnn')
    params = data.get('params', {})
    
    if not filename or not target_column:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Read data
        df = pd.read_csv(filepath)
        
        # Start timing
        start_time = time.time()
        
        # Train appropriate model
        if model_type == 'cnn':
            y_test, y_pred, history = train_cnn_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'CNN Model: Actual vs Predicted {target_column}')
            
            # Prepare history data for JSON
            history_dict = {
                'loss': [float(val) for val in history.history['loss']],
                'val_loss': [float(val) for val in history.history['val_loss']]
}
            
            return jsonify({
                'model': 'CNN',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'history': history_dict,
                'trainingTime': training_time
            })
            
        elif model_type == 'lstm':
            y_test, y_pred, history = train_lstm_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'LSTM Model: Actual vs Predicted {target_column}')
            
            # Prepare history data for JSON
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            return jsonify({
                'model': 'LSTM',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'history': history_dict,
                'trainingTime': training_time
            })
            
        elif model_type == 'arima':
            y_test, y_pred, aic = train_arima_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'ARIMA Model: Actual vs Predicted {target_column}')
            
            return jsonify({
                'model': 'ARIMA',
                'mse': float(mse),
                'rmse': float(rmse),
                'aic': float(aic),
                'plot': plot,
                'trainingTime': training_time
            })
            
        elif model_type == 'prophet':
            y_test, y_pred, mape = train_prophet_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'Prophet Model: Actual vs Predicted {target_column}')
            
            return jsonify({
                'model': 'Prophet',
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'plot': plot,
                'trainingTime': training_time
            })
            
        elif model_type == 'xgboost':
            y_test, y_pred, feature_importance, feature_names, training_time = train_xgboost_model(df, target_column, params)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plots
            plot = create_plot(y_test, y_pred, f'XGBoost Model: Actual vs Predicted {target_column}')
            feature_plot = create_feature_importance_plot(feature_names, feature_importance)
            
            return jsonify({
                'model': 'XGBoost',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'featureImportance': feature_plot,
                'trainingTime': training_time
            })
            
        elif model_type == 'transformer':
            y_test, y_pred, history = train_transformer_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'Transformer Model: Actual vs Predicted {target_column}')
            
            # Prepare history data for JSON
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            return jsonify({
                'model': 'Transformer',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'history': history_dict,
                'trainingTime': training_time
            })
        # train() fonksiyonunda yeni modellerin işlenmesi için eklenecek kod parçaları

# SARIMA modeli için eklenecek elif bloğu:
        elif model_type == 'sarima':
            y_test, y_pred, aic = train_sarima_model(df, target_column, params)
            training_time = time.time() - start_time
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'SARIMA Model: Actual vs Predicted {target_column}')
            
            return jsonify({
                'model': 'SARIMA',
                'mse': float(mse),
                'rmse': float(rmse),
                'aic': float(aic),
                'plot': plot,
                'trainingTime': training_time
            })

# Random Forest modeli için eklenecek elif bloğu:
        elif model_type == 'random_forest':
            y_test, y_pred, feature_importance, feature_names, training_time = train_random_forest_model(df, target_column, params)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plots
            plot = create_plot(y_test, y_pred, f'Random Forest Model: Actual vs Predicted {target_column}')
            feature_plot = create_feature_importance_plot(feature_names, feature_importance)
            
            return jsonify({
                'model': 'Random Forest',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'featureImportance': feature_plot,
                'trainingTime': training_time
            })

        # Rocket modeli için eklenecek elif bloğu:
        elif model_type == 'rocket':
            y_test, y_pred, accuracy, training_time = train_rocket_model(df, target_column, params)
            
            # Calculate metrics (classification metrics)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            try:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            except:
                precision = recall = f1 = 0.0
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'ROCKET Model: Actual vs Predicted {target_column}')
            
            return jsonify({
                'model': 'ROCKET',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'plot': plot,
                'trainingTime': training_time
            })

        # InceptionTime modeli için eklenecek elif bloğu:
        elif model_type == 'inception_time':
            y_test, y_pred, history, training_time = train_inception_time_model(df, target_column, params)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'InceptionTime Model: Actual vs Predicted {target_column}')
            
            # Prepare history data for JSON
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            return jsonify({
                'model': 'InceptionTime',
                'mse': float(mse),
                'rmse': float(rmse),
                'plot': plot,
                'history': history_dict,
                'trainingTime': training_time
            })

        # Shapelet Transform modeli için eklenecek elif bloğu:
        elif model_type == 'shapelet_transform':
            y_test, y_pred, accuracy, training_time = train_shapelet_transform_model(df, target_column, params)
            
            # Calculate metrics (classification metrics)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            try:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
            except:
                precision = recall = f1 = 0.0
            
            # Create plot
            plot = create_plot(y_test, y_pred, f'Shapelet Transform Model: Actual vs Predicted {target_column}')
            
            return jsonify({
                'model': 'Shapelet Transform',
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'plot': plot,
                'trainingTime': training_time
            })


            
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)