import zipfile
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import h2o
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Load the saved H2O model
model_path = r"C:\Users\MAHARAJ\OneDrive\Desktop\P\saved_model\DRF_model_python_1751807635499_1"
loaded_model = h2o.load_model(model_path)

# Define paths for saving datasets and results
data_folder = r"C:\Users\MAHARAJ\OneDrive\Desktop\P"
dataset_path = os.path.join(data_folder, "2020.06.19.csv")
train_data_path = os.path.join(data_folder, "train.csv")
test_data_path = os.path.join(data_folder, "test.csv")
results_folder = r"C:\Users\MAHARAJ\OneDrive\Desktop\P\results"

# Ensure results directory exists
os.makedirs(results_folder, exist_ok=True)

# Email configuration
email_address = "manaswinikhanna23@gmail.com"
email_password = "vgux xzck xovb zfgj"

# Function to split and save dataset
def split_and_save_dataset(data_path, train_path, test_path, test_size=0.2):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

# Function to make predictions and save results as CSV (no swaps)
def make_predictions_and_save_results(test_df):
    test_df_hex = h2o.H2OFrame(test_df)
    predictions = loaded_model.predict(test_df_hex)
    predictions_df = predictions.as_data_frame()
    results_df = pd.concat([test_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file_path = os.path.join(results_folder, f"prediction_results_{timestamp_str}.csv")
    results_df.to_csv(results_file_path, index=False)
    return results_file_path

# Function to send email with attachment
def send_email_with_attachment(receiver_email, subject, body, file_path):
    sender_email = email_address
    sender_password = email_password
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    compressed_file_path = compress_file(file_path)
    with open(compressed_file_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(compressed_file_path)}")
    message.attach(part)
    text = message.as_string()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, text)
        print(f"Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Function to compress file
def compress_file(file_path):
    zip_file_path = file_path + '.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path), compress_type=zipfile.ZIP_DEFLATED)
    return zip_file_path

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    if username == "manaswinikhanna23@gmail.com" and password == "1234":
        return redirect(url_for('upload_image'))
    else:
        return "Invalid credentials. Please try again."

@app.route('/upload_image')
def upload_image():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file.filename == '':
                return "No file selected. Please choose a file."
            uploaded_file_path = os.path.join(data_folder, file.filename)
            file.save(uploaded_file_path)
            split_and_save_dataset(uploaded_file_path, train_data_path, test_data_path)
            test_df = pd.read_csv(test_data_path)
            if 'dest_port' in test_df.columns:
                test_df['dest_port'] = test_df['dest_port'].fillna(-1).astype('int64')
            if 'src_port' in test_df.columns:
                test_df['src_port'] = test_df['src_port'].fillna(-1).astype('int64')
            results_file_path = make_predictions_and_save_results(test_df)
            send_email_with_attachment(
                "manaswinikhanna23@gmail.com",
                "Network Intrusion Detection - Prediction Results",
                f"Please find the prediction results attached.\n\nFile: {os.path.basename(results_file_path)}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                results_file_path
            )
            prediction_results = pd.read_csv(results_file_path).to_dict(orient='records')
            results_df = pd.read_csv(results_file_path)
            benign_count = len(results_df[results_df['predict'] == 'benign'])
            malicious_count = len(results_df[results_df['predict'] == 'malicious'])
            outlier_count = len(results_df[results_df['predict'] == 'outlier'])
            total_count = len(results_df)
            message = f"""
            Prediction completed successfully!
            
            Results Summary:
            - Total records processed: {total_count}
            - Benign predictions: {benign_count}
            - Malicious predictions: {malicious_count}
            - Outlier predictions: {outlier_count}
            
            Results saved at: {results_file_path}
            Email sent to: manaswinikhanna23@gmail.com
            """
            return render_template('result.html', message=message, prediction_results=prediction_results[:100])
        except Exception as e:
            error_message = f"An error occurred during prediction: {str(e)}"
            return render_template('result.html', message=error_message, prediction_results=[])

@app.route('/process_default_dataset')
def process_default_dataset():
    try:
        if not os.path.exists(dataset_path):
            error_message = f"Default dataset not found at: {dataset_path}"
            return render_template('result.html', message=error_message, prediction_results=[])
        split_and_save_dataset(dataset_path, train_data_path, test_data_path)
        test_df = pd.read_csv(test_data_path)
        if 'dest_port' in test_df.columns:
            test_df['dest_port'] = test_df['dest_port'].fillna(-1).astype('int64')
        if 'src_port' in test_df.columns:
            test_df['src_port'] = test_df['src_port'].fillna(-1).astype('int64')
        results_file_path = make_predictions_and_save_results(test_df)
        send_email_with_attachment(
            "manaswinikhanna23@gmail.com",
            "Network Intrusion Detection - Default Dataset Results",
            f"Please find the prediction results for the default dataset attached.\n\nFile: {os.path.basename(results_file_path)}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            results_file_path
        )
        prediction_results = pd.read_csv(results_file_path).to_dict(orient='records')
        results_df = pd.read_csv(results_file_path)
        benign_count = len(results_df[results_df['predict'] == 'benign'])
        malicious_count = len(results_df[results_df['predict'] == 'malicious'])
        outlier_count = len(results_df[results_df['predict'] == 'outlier'])
        total_count = len(results_df)
        message = f"""
        Default dataset prediction completed successfully!
        
        Results Summary:
        - Total records processed: {total_count}
        - Benign predictions: {benign_count}
        - Malicious predictions: {malicious_count}
        - Outlier predictions: {outlier_count}
        
        Results saved at: {results_file_path}
        Email sent to: manaswinikhanna23@gmail.com
        """
        return render_template('result.html', message=message, prediction_results=prediction_results[:100])
    except Exception as e:
        error_message = f"An error occurred while processing default dataset: {str(e)}"
        return render_template('result.html', message=error_message, prediction_results=[])

if __name__ == '__main__':
    print("Starting Network Intrusion Detection Flask App...")
    print("Make sure H2O is properly initialized and the model file exists.")
    app.run(debug=True, host='0.0.0.0', port=5000)
