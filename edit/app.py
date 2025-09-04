from flask import Flask, request, render_template
import pandas as pd
import h2o

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Load the saved H2O model
model_path = "C:/Users/nithi/OneDrive/Desktop/dhurva/saved_model/DRF_model_python_1713967822737_1"
loaded_model = h2o.load_model(model_path)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']

        try:
            # Read uploaded file into pandas DataFrame with appropriate encoding
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            # If 'utf-8' encoding fails, try 'latin1' (ISO-8859-1) encoding
            df = pd.read_csv(file, encoding='latin1')

        # Convert DataFrame to H2OFrame
        df_hex = h2o.H2OFrame(df)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(df_hex)

        # Convert predictions to pandas DataFrame
        predictions_df = predictions.as_data_frame()

        # Extract the 'predict' column from the DataFrame as a list
        prediction_results = predictions_df['predict'].tolist()

        # Prepare prediction results to pass to template
        return render_template('result.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
