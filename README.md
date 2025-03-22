# timeseries_toolbox
## Running the Application


1. Create a virtual environment (optional but recommended):
```bash
cd timeseries-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```


3. Install required packages for frontend:
```bash
npm install axios
```


4. Start the backend server:
```bash
cd backend
python app.py
```


5. In a new terminal, start the frontend:
```bash
cd frontend
npm start
```

6. Open your browser and navigate to http://localhost:3000

## Usage Instructions

1. Upload a CSV file containing time series data
2. Select the target column you want to predict
3. Choose between  timeseries models
4. Configure model parameters
5. Click "Train Model" to train and evaluate the model
6. View results including MSE, RMSE, and prediction plots
