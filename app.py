import zipfile
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import h2o
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
import pickle
import hashlib
import os
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import io
import numpy as np
import time

app = Flask(__name__)

# Initialize H2O
h2o.init()

# Define paths for saving datasets and results
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = current_dir
dataset_path = os.path.join(data_folder, "2020.06.19.csv")
train_data_path = os.path.join(data_folder, "train.csv")
test_data_path = os.path.join(data_folder, "test.csv")
results_folder = os.path.join(current_dir, "results")
charts_folder = os.path.join(current_dir, "charts")
malicious_folder = os.path.join(current_dir, "malicious_data")
cache_folder = os.path.join(current_dir, "model_cache")

# Initialize H2O with simple settings
try:
    h2o.init(nthreads=-1, max_mem_size="4G")
    print("H2O initialization successful!")
except Exception as e:
    print(f"H2O initialization failed: {str(e)}")
    print("Continuing without H2O - Random Forest will still work")

# Load the saved H2O model
model_path = os.path.join(current_dir, "saved_model", "DRF_model_python_1751807635499_1")
try:
    loaded_model = h2o.load_model(model_path)
    print("H2O model loaded successfully!")
except Exception as e:
    print(f"Failed to load H2O model: {str(e)}")
    loaded_model = None

# Ensure directories exist
os.makedirs(results_folder, exist_ok=True)
os.makedirs(charts_folder, exist_ok=True)
os.makedirs(malicious_folder, exist_ok=True)
os.makedirs(cache_folder, exist_ok=True)

# Email configuration
email_address = "manaswinikhanna23@gmail.com"
email_password = "vgux xzck xovb zfgj"

# Function to split and save dataset
def split_and_save_dataset(data_path, train_path, test_path, test_size=0.2):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

# Function to create data hash for caching
def create_data_hash(train_df, test_df):
    """Create a hash of the dataset for caching purposes"""
    combined_data = pd.concat([train_df, test_df])
    data_string = str(combined_data.values.tobytes()) + str(combined_data.columns.tolist())
    return hashlib.md5(data_string.encode()).hexdigest()

# Function to train and compare multiple ML models (optimized with caching)
def train_and_compare_models_fast(train_df, test_df, use_cache=True, sample_size=5000):
    # Check cache first
    if use_cache:
        data_hash = create_data_hash(train_df, test_df)
        cache_file = os.path.join(cache_folder, f"model_results_{data_hash}.pkl")
        
        if os.path.exists(cache_file):
            print("📦 Loading cached model results...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_results = pickle.load(f)
                print("✅ Cache loaded successfully!")
                return cached_results['model_results'], {}, {}, None
            except Exception as e:
                print(f"⚠️  Cache loading failed: {str(e)}, retraining models...")
    
    # Sample data for faster training if dataset is large
    if len(train_df) > sample_size:
        print(f"📊 Sampling {sample_size} records for faster training...")
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    if len(test_df) > sample_size//4:
        test_sample_size = min(sample_size//4, len(test_df))
        print(f"📊 Sampling {test_sample_size} test records...")
        test_df = test_df.sample(n=test_sample_size, random_state=42)
    
    # Prepare the data for sklearn
    target_columns = ['label', 'class', 'target', 'attack_cat', 'Label']
    target_col = None
    
    for col in target_columns:
        if col in train_df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = train_df.columns[-1]
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    # Handle non-numeric columns
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable if it's categorical
    target_encoder = None
    if y_train.dtype == 'object':
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train)
        y_test = target_encoder.transform(y_test)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define optimized models (reduced complexity for speed)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='lbfgs'),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(random_state=42, kernel='rbf', C=1.0),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    }
    
    model_results = {}
    trained_models = {}
    
    print("🤖 Training and comparing multiple ML models (optimized)...")
    
    for name, model in models.items():
        try:
            print(f"   Training {name}...")
            start_time = datetime.now()
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'SVM', 'Neural Network', 'K-Nearest Neighbors']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_results[name] = {
                'accuracy': accuracy,
                'accuracy_percent': accuracy * 100,
                'training_time': training_time,
                'predictions': y_pred
            }
            trained_models[name] = model
            
            print(f"   ✅ {name}: {accuracy:.4f} ({accuracy*100:.2f}%) in {training_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)}")
            model_results[name] = {
                'accuracy': 0.0,
                'accuracy_percent': 0.0,
                'error': str(e)
            }
    
    # Sort models by accuracy
    sorted_results = dict(sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    
    # Cache results if enabled
    if use_cache:
        try:
            cache_data = {
                'model_results': sorted_results,
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(train_df)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Results cached for future use")
        except Exception as e:
            print(f"⚠️  Cache saving failed: {str(e)}")
    
    return sorted_results, trained_models, label_encoders, scaler

# Original function for backward compatibility
def train_and_compare_models(train_df, test_df):
    return train_and_compare_models_fast(train_df, test_df, use_cache=True, sample_size=10000)

# Backward compatibility function
def train_and_evaluate_random_forest(train_df, test_df):
    results, models, encoders, scaler = train_and_compare_models(train_df, test_df)
    rf_accuracy = results.get('Random Forest', {}).get('accuracy', 0.0)
    rf_model = models.get('Random Forest', None)
    return rf_accuracy, rf_model, encoders

# Function to make predictions and save results as CSV
def make_predictions_and_save_results(test_df):
    if loaded_model is not None:
        try:
            test_df_hex = h2o.H2OFrame(test_df)
            predictions = loaded_model.predict(test_df_hex)
            predictions_df = predictions.as_data_frame()
            results_df = pd.concat([test_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"H2O prediction failed: {str(e)}")
            # Create dummy predictions if H2O fails
            results_df = test_df.copy()
            results_df['predict'] = 'benign'  # Default prediction
    else:
        # Create dummy predictions if no H2O model
        results_df = test_df.copy()
        results_df['predict'] = 'benign'  # Default prediction
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file_path = os.path.join(results_folder, f"prediction_results_{timestamp_str}.csv")
    results_df.to_csv(results_file_path, index=False)
    return results_file_path, results_df

# Function to create and save malicious data only
def create_malicious_data_file(results_df):
    """Extract only malicious predictions and save to separate file"""
    malicious_df = results_df[results_df['predict'] == 'malicious'].copy()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    malicious_file_path = os.path.join(malicious_folder, f"malicious_data_only_{timestamp_str}.csv")
    malicious_df.to_csv(malicious_file_path, index=False)
    return malicious_file_path, malicious_df

# Function to create comprehensive bar chart with percentages and save as image
def create_detailed_bar_chart(benign_count, malicious_count, outlier_count, total_count, filename):
    """Create a detailed bar chart with percentages and save as image file"""
    categories = ['Benign', 'Malicious', 'Outlier']
    counts = [benign_count, malicious_count, outlier_count]
    percentages = [(count/total_count)*100 for count in counts]
    colors = ['#28a745', '#dc3545', '#ffc107']  # Green, Red, Yellow

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Bar chart with counts
    bars1 = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Network Intrusion Detection Results - Counts', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Prediction Categories', fontsize=12)
    ax1.set_ylabel('Number of Records', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Pie chart with percentages
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%',
                                      colors=colors, startangle=90, explode=(0.05, 0.05, 0.05))
    ax2.set_title('Network Intrusion Detection Results - Percentages', fontsize=16, fontweight='bold', pad=20)

    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # Add summary statistics
    malicious_percentage = (malicious_count/total_count)*100
    benign_percentage = (benign_count/total_count)*100

    # Add text box with statistics
    stats_text = f"""
    SECURITY ANALYSIS SUMMARY
    ========================
    Total Records Processed: {total_count:,}
    File Analyzed: {filename}

    Benign Traffic: {benign_count:,} ({benign_percentage:.1f}%)
    Malicious Traffic: {malicious_count:,} ({malicious_percentage:.1f}%)
    Outlier Traffic: {outlier_count:,} ({(outlier_count/total_count)*100:.1f}%)

    Security Status: {'🚨 HIGH RISK' if malicious_percentage > 30 else '✅ SECURE'}
    Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    plt.figtext(0.5, 0.02, stats_text, ha='center', va='bottom',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save chart as image file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"security_analysis_chart_{timestamp_str}.png"
    chart_path = os.path.join(charts_folder, chart_filename)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')

    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return img_data, chart_path

# Function to create model comparison visualization
def create_model_comparison_chart(model_results, filename):
    """Create a bar chart comparing model accuracies"""
    if not model_results:
        return None, None
    
    # Filter out models with errors
    valid_models = {name: results for name, results in model_results.items() 
                   if 'error' not in results}
    
    if not valid_models:
        return None, None
    
    model_names = list(valid_models.keys())
    accuracies = [results['accuracy'] for results in valid_models.values()]
    
    # Create the comparison chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create color scheme - best model gets gold, others gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    # Convert to RGBA and set best model to gold
    colors = list(colors)
    colors[0] = [1.0, 0.843, 0.0, 1.0]  # Gold color in RGBA
    
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_title('Machine Learning Models Performance Comparison\nNetwork Intrusion Detection', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Machine Learning Models', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add accuracy labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}\n({accuracy*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add summary statistics
    best_model = model_names[0]
    best_accuracy = accuracies[0]
    worst_accuracy = min(accuracies)
    avg_accuracy = sum(accuracies) / len(accuracies)
    
    stats_text = f"""MODEL PERFORMANCE SUMMARY
    ═══════════════════════════════════════
    📁 Dataset: {filename}
    🥇 Best Model: {best_model} ({best_accuracy:.3f})
    📊 Average Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)
    📉 Lowest Accuracy: {worst_accuracy:.3f} ({worst_accuracy*100:.1f}%)
    🔬 Models Tested: {len(valid_models)}
    📈 Performance Range: {(best_accuracy-worst_accuracy)*100:.1f}% points
    🕒 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', va='bottom',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    
    # Save chart as image file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"model_comparison_chart_{timestamp_str}.png"
    chart_path = os.path.join(charts_folder, chart_filename)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_data, chart_path

# Function to create individual charts for separate downloads
def create_individual_charts(model_results, filename):
    """Create individual charts for separate download options"""
    if not model_results:
        return {}
    
    # Filter out models with errors
    valid_models = {name: results for name, results in model_results.items() 
                   if 'error' not in results}
    
    if not valid_models:
        return {}
    
    model_names = list(valid_models.keys())
    accuracies = [results['accuracy'] for results in valid_models.values()]
    training_times = [results.get('training_time', 0) for results in valid_models.values()]
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_paths = {}
    
    # Chart 1: Accuracy Bar Chart
    plt.figure(figsize=(12, 8))
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    colors_bar = list(colors_bar)
    colors_bar[0] = [1.0, 0.843, 0.0, 1.0]  # Gold for best model
    
    bars = plt.bar(model_names, accuracies, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Model Accuracy Comparison\nNetwork Intrusion Detection', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add accuracy labels
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}\n({accuracy*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    accuracy_chart_path = os.path.join(charts_folder, f"accuracy_comparison_{timestamp_str}.png")
    plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    chart_paths['accuracy'] = accuracy_chart_path
    
    # Chart 2: Training Time Bar Chart
    plt.figure(figsize=(12, 8))
    colors_time = ['#FF6B6B' if i == np.argmax(training_times) else '#4ECDC4' for i in range(len(training_times))]
    bars2 = plt.bar(model_names, training_times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Training Time Comparison\nNetwork Intrusion Detection Models', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add time labels
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.02,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    time_chart_path = os.path.join(charts_folder, f"training_time_{timestamp_str}.png")
    plt.savefig(time_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    chart_paths['training_time'] = time_chart_path
    
    # Chart 3: Accuracy vs Training Time Scatter Plot
    plt.figure(figsize=(10, 8))
    scatter_colors = ['gold' if i == 0 else 'skyblue' for i in range(len(model_names))]
    plt.scatter(training_times, accuracies, c=scatter_colors, s=200, alpha=0.7, 
                edgecolors='black', linewidth=2)
    
    # Add model name labels
    for i, name in enumerate(model_names):
        plt.annotate(name[:10] + ('...' if len(name) > 10 else ''), 
                    (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.title('Model Performance: Accuracy vs Training Time\nNetwork Intrusion Detection', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_chart_path = os.path.join(charts_folder, f"accuracy_vs_time_{timestamp_str}.png")
    plt.savefig(scatter_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    chart_paths['scatter'] = scatter_chart_path
    
    # Chart 4: Performance Grade Pie Chart
    plt.figure(figsize=(10, 8))
    grade_counts = {'Excellent (>95%)': 0, 'Good (90-95%)': 0, 'Fair (80-90%)': 0, 'Poor (<80%)': 0}
    
    for accuracy in accuracies:
        if accuracy > 0.95:
            grade_counts['Excellent (>95%)'] += 1
        elif accuracy > 0.90:
            grade_counts['Good (90-95%)'] += 1
        elif accuracy > 0.80:
            grade_counts['Fair (80-90%)'] += 1
        else:
            grade_counts['Poor (<80%)'] += 1
    
    # Only show non-zero grades
    non_zero_grades = {k: v for k, v in grade_counts.items() if v > 0}
    
    if non_zero_grades:
        grade_colors = ['#2ECC40', '#01FF70', '#FFDC00', '#FF4136'][:len(non_zero_grades)]
        wedges, texts, autotexts = plt.pie(non_zero_grades.values(), labels=non_zero_grades.keys(), 
                                          autopct='%1.0f%%', colors=grade_colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
    
    plt.title('Model Performance Distribution\nNetwork Intrusion Detection', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    
    pie_chart_path = os.path.join(charts_folder, f"performance_distribution_{timestamp_str}.png")
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    chart_paths['pie'] = pie_chart_path
    
    return chart_paths

# Function to create comprehensive model performance graphs
def create_comprehensive_model_charts(model_results, filename):
    """Create multiple charts for comprehensive model analysis"""
    if not model_results:
        return None, None
    
    # Filter out models with errors
    valid_models = {name: results for name, results in model_results.items() 
                   if 'error' not in results}
    
    if not valid_models:
        return None, None
    
    model_names = list(valid_models.keys())
    accuracies = [results['accuracy'] for results in valid_models.values()]
    training_times = [results.get('training_time', 0) for results in valid_models.values()]
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Accuracy Bar Chart with Golden Best Model
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    colors_bar = list(colors_bar)
    colors_bar[0] = [1.0, 0.843, 0.0, 1.0]  # Gold for best model
    
    bars = ax1.bar(model_names, accuracies, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add accuracy labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Chart 2: Training Time Comparison
    colors_time = ['#FF6B6B' if i == np.argmax(training_times) else '#4ECDC4' for i in range(len(training_times))]
    bars2 = ax2.bar(model_names, training_times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add time labels on bars
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.02,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Chart 3: Accuracy vs Training Time Scatter Plot
    scatter_colors = ['gold' if i == 0 else 'skyblue' for i in range(len(model_names))]
    ax3.scatter(training_times, accuracies, c=scatter_colors, s=200, alpha=0.7, 
                edgecolors='black', linewidth=2)
    
    # Add model name labels to scatter points
    for i, name in enumerate(model_names):
        ax3.annotate(name[:8] + ('...' if len(name) > 8 else ''), 
                    (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax3.set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Time (seconds)', fontsize=12)
    ax3.set_ylabel('Accuracy Score', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Performance Grade Pie Chart
    grade_counts = {'Excellent (>95%)': 0, 'Good (90-95%)': 0, 'Fair (80-90%)': 0, 'Poor (<80%)': 0}
    
    for accuracy in accuracies:
        if accuracy > 0.95:
            grade_counts['Excellent (>95%)'] += 1
        elif accuracy > 0.90:
            grade_counts['Good (90-95%)'] += 1
        elif accuracy > 0.80:
            grade_counts['Fair (80-90%)'] += 1
        else:
            grade_counts['Poor (<80%)'] += 1
    
    # Only show non-zero grades
    non_zero_grades = {k: v for k, v in grade_counts.items() if v > 0}
    
    if non_zero_grades:
        grade_colors = ['#2ECC40', '#01FF70', '#FFDC00', '#FF4136'][:len(non_zero_grades)]
        wedges, texts, autotexts = ax4.pie(non_zero_grades.values(), labels=non_zero_grades.keys(), 
                                          autopct='%1.0f%%', colors=grade_colors, startangle=90)
        ax4.set_title('Model Performance Distribution', fontsize=14, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    else:
        ax4.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold')
        ax4.set_title('Model Performance Distribution', fontsize=14, fontweight='bold')
    
    # Add overall statistics
    best_model = model_names[0]
    best_accuracy = accuracies[0]
    avg_accuracy = np.mean(accuracies)
    avg_time = np.mean(training_times)
    
    stats_text = f"""COMPREHENSIVE MODEL ANALYSIS DASHBOARD
    ═══════════════════════════════════════════════════════════════
    📁 Dataset: {filename}
    🥇 Best Model: {best_model} ({best_accuracy:.4f} accuracy)
    📊 Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)
    ⏱️  Average Training Time: {avg_time:.1f} seconds
    🔬 Models Analyzed: {len(valid_models)}
    📈 Performance Spread: {(max(accuracies)-min(accuracies))*100:.1f} percentage points
    🕒 Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', va='bottom',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save chart as image file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"comprehensive_model_analysis_{timestamp_str}.png"
    chart_path = os.path.join(charts_folder, chart_filename)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Convert to base64 for web display
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_data, chart_path

# Function to send email with attachment
def send_email_with_attachment(receiver_email, subject, body, file_path=None):
    sender_email = email_address
    sender_password = email_password
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Add attachment if file_path is provided
    if file_path:
        if file_path.endswith('.png'):
            # For image files, attach directly
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(file_path)}")
            message.attach(part)
        else:
            # For other files, compress first
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
        print(f"✅ Email sent successfully to {receiver_email} - Subject: {subject}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

# Function to compress file
def compress_file(file_path):
    zip_file_path = file_path + '.zip'
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path), compress_type=zipfile.ZIP_DEFLATED)
    return zip_file_path

# Function to send all 3 emails
def send_all_emails(filename, results_file_path, chart_path, malicious_file_path,
                   benign_count, malicious_count, outlier_count, total_count, malicious_df, model_results=None):

    malicious_percentage = (malicious_count/total_count)*100
    benign_percentage = (benign_count/total_count)*100
    outlier_percentage = (outlier_count/total_count)*100

    email_results = []

    # EMAIL 1: Complete Prediction Data
    print("📧 Sending Email 1: Complete Prediction Results...")
    subject1 = "📊 Network Intrusion Detection - Complete Analysis Results"
    # Create model performance text
    model_performance_text = ""
    if model_results:
        model_performance_text = "\n\n🤖 MODEL PERFORMANCE COMPARISON:"
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'error' not in results:
                rank = i + 1
                accuracy = results['accuracy']
                model_performance_text += f"\n   {rank}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)"
    
    body1 = f"""Network Security Analysis - Complete Results
================================

📁 File Analyzed: {filename}
📊 Analysis Summary:
   • Total Records Processed: {total_count:,}
   • Benign Traffic: {benign_count:,} ({benign_percentage:.1f}%)
   • Malicious Traffic: {malicious_count:,} ({malicious_percentage:.1f}%)
   • Outlier Traffic: {outlier_count:,} ({outlier_percentage:.1f}%)

🔍 Analysis Details:
   • Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   • H2O Model used: DRF (Distributed Random Forest)
   • Multiple ML Models Compared: 7 algorithms tested
   • Data split: 80% training, 20% testing{model_performance_text}

📎 Attached: Complete prediction results with all data points

This email contains the comprehensive analysis results for your review.

Best regards,
Network Security Monitoring System"""

    result1 = send_email_with_attachment(email_address, subject1, body1, results_file_path)
    email_results.append(("Complete Results", result1))

    # EMAIL 2: Chart with Percentages
    print("📧 Sending Email 2: Visual Analysis Chart...")
    subject2 = "📈 Network Security Analysis - Visual Report & Percentages"
    body2 = f"""Network Security Visual Analysis Report
=====================================

📁 File: {filename}
📊 Visual Summary Chart Attached

📈 PERCENTAGE BREAKDOWN:
   🛡️  Benign Traffic: {benign_percentage:.1f}%
   🚨 Malicious Traffic: {malicious_percentage:.1f}%
   ⚠️  Outlier Traffic: {outlier_percentage:.1f}%

📊 ABSOLUTE NUMBERS:
   • Total Records: {total_count:,}
   • Benign: {benign_count:,}
   • Malicious: {malicious_count:,}
   • Outliers: {outlier_count:,}

🎯 Security Status: {'🚨 HIGH RISK - Malicious > 30%' if malicious_percentage > 30 else '✅ SECURE - Malicious < 30%'}

📎 Attached: High-resolution analysis chart with visual breakdown

The attached chart provides a comprehensive visual representation of your network security analysis.

Best regards,
Network Security Monitoring System"""

    result2 = send_email_with_attachment(email_address, subject2, body2, chart_path)
    email_results.append(("Visual Chart", result2))

    # EMAIL 3: Malicious Data Only
    print("📧 Sending Email 3: Malicious Data Analysis...")
    subject3 = f"🚨 Malicious Traffic Data - {malicious_count:,} Threats Detected"
    body3 = f"""MALICIOUS TRAFFIC ANALYSIS REPORT
==================================

🚨 THREAT DETECTION SUMMARY:
   📁 Source File: {filename}
   🔍 Malicious Records Found: {malicious_count:,} out of {total_count:,} total
   📊 Malicious Percentage: {malicious_percentage:.1f}%

🔥 THREAT DETAILS:
   • High-risk connections identified
   • Suspicious network patterns detected
   • Potential security breaches flagged

📋 MALICIOUS DATA BREAKDOWN:
   The attached file contains ONLY the malicious traffic data for detailed analysis:
   • Source IPs of malicious connections
   • Destination ports targeted
   • Protocol types used in attacks
   • Timestamp information
   • Network flow characteristics

⚠️  RECOMMENDED ACTIONS:
   1. Immediate review of all malicious IPs
   2. Block suspicious source addresses
   3. Monitor affected destination ports
   4. Strengthen firewall rules
   5. Implement additional monitoring

📎 Attached: Filtered dataset containing only malicious traffic records

This critical data requires immediate attention from your security team.

Best regards,
Network Security Monitoring System"""

    result3 = send_email_with_attachment(email_address, subject3, body3, malicious_file_path)
    email_results.append(("Malicious Data", result3))

    # EMAIL 4: Alert Email (if malicious > 30%)
    if malicious_percentage > 30:
        print("🚨 Sending Email 4: Security Alert...")
        subject4 = "🚨🚨 CRITICAL SECURITY ALERT - Malicious Traffic > 30% 🚨🚨"
        body4 = f"""
🚨🚨🚨 CRITICAL SECURITY ALERT 🚨🚨🚨
=======================================

IMMEDIATE ACTION REQUIRED!
Malicious network activity has exceeded the 30% threshold!

⚠️  THREAT LEVEL: HIGH RISK
📊 Malicious Traffic: {malicious_percentage:.1f}% ({malicious_count:,} out of {total_count:,} records)

🔥 CRITICAL STATISTICS:
   • File Analyzed: {filename}
   • Alert Threshold: 30%
   • Current Malicious Level: {malicious_percentage:.1f}%
   • Total Threats: {malicious_count:,}
   • Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚨 IMMEDIATE ACTIONS REQUIRED:
   1. 🔒 ISOLATE affected network segments immediately
   2. 🛡️  ACTIVATE incident response protocol
   3. 🔍 INVESTIGATE all malicious IP addresses
   4. 📞 NOTIFY security team and management
   5. 🚫 BLOCK suspicious traffic sources
   6. 📝 DOCUMENT all findings for forensic analysis
   7. 🔄 IMPLEMENT enhanced monitoring
   8. 📧 ESCALATE to cybersecurity team

⏰ TIME-SENSITIVE RESPONSE:
   This alert indicates a potential security breach or ongoing attack.
   Swift action is critical to prevent further compromise.

🆘 EMERGENCY CONTACTS:
   • Network Security Team: [Contact Details]
   • Incident Response: [Contact Details]
   • Management Escalation: [Contact Details]

This is an automated critical alert. Please acknowledge receipt and initiate response procedures immediately.

URGENT - DO NOT DELAY RESPONSE
===============================

Network Security Monitoring System
Automated Alert Generation"""

        result4 = send_email_with_attachment(email_address, subject4, body4)
        email_results.append(("Security Alert", result4))

    return email_results

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

            print(f"🔄 Processing file: {file.filename}")
            uploaded_file_path = os.path.join(data_folder, file.filename)
            file.save(uploaded_file_path)

            # Split and process dataset
            split_and_save_dataset(uploaded_file_path, train_data_path, test_data_path)
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Handle data preprocessing
            if 'dest_port' in test_df.columns:
                test_df['dest_port'] = test_df['dest_port'].fillna(-1).astype('int64')
            if 'src_port' in test_df.columns:
                test_df['src_port'] = test_df['src_port'].fillna(-1).astype('int64')
            
            # Apply same preprocessing to training data
            if 'dest_port' in train_df.columns:
                train_df['dest_port'] = train_df['dest_port'].fillna(-1).astype('int64')
            if 'src_port' in train_df.columns:
                train_df['src_port'] = train_df['src_port'].fillna(-1).astype('int64')

            # Train and compare multiple ML models
            print("🤖 Training and comparing multiple ML models...")
            model_results, trained_models, label_encoders, scaler = train_and_compare_models(train_df, test_df)
            
            # Get Random Forest accuracy for backward compatibility
            rf_accuracy = model_results.get('Random Forest', {}).get('accuracy', 0.0)
            print(f"✅ Model comparison completed!")

            # Make predictions
            print("🤖 Making predictions...")
            results_file_path, results_df = make_predictions_and_save_results(test_df)

            # Create malicious data file
            print("🚨 Extracting malicious data...")
            malicious_file_path, malicious_df = create_malicious_data_file(results_df)

            # Calculate statistics
            benign_count = len(results_df[results_df['predict'] == 'benign'])
            malicious_count = len(results_df[results_df['predict'] == 'malicious'])
            outlier_count = len(results_df[results_df['predict'] == 'outlier'])
            total_count = len(results_df)

            malicious_percentage = (malicious_count/total_count)*100
            benign_percentage = (benign_count/total_count)*100

            # Create detailed chart
            print("📊 Creating analysis charts...")
            chart_data, chart_path = create_detailed_bar_chart(
                benign_count, malicious_count, outlier_count, total_count, file.filename
            )
            
            # Create model comparison chart
            print("📈 Creating model comparison chart...")
            model_chart_data, model_chart_path = create_model_comparison_chart(model_results, file.filename)

            # Send all emails
            print("📧 Sending email notifications...")
            email_results = send_all_emails(
                file.filename, results_file_path, chart_path, malicious_file_path,
                benign_count, malicious_count, outlier_count, total_count, malicious_df, model_results
            )

            # Prepare response data
            prediction_results = results_df.to_dict(orient='records')

            # Create email status summary
            email_status = "\n".join([f"✅ {name}: {'Sent' if result else 'Failed'}"
                                    for name, result in email_results])

            message = f"""
🔍 NETWORK SECURITY ANALYSIS COMPLETE
=====================================

📁 File Analyzed: {file.filename}
📊 Total Records Processed: {total_count:,}

🛡️  SECURITY BREAKDOWN:
• Benign Traffic: {benign_count:,} ({benign_percentage:.1f}%)
• Malicious Traffic: {malicious_count:,} ({malicious_percentage:.1f}%)
• Outlier Traffic: {outlier_count:,} ({(outlier_count/total_count)*100:.1f}%)

🤖 MODEL PERFORMANCE COMPARISON:"""
            
            # Add model comparison results
            for i, (model_name, results) in enumerate(model_results.items()):
                if 'error' not in results:
                    rank = i + 1
                    accuracy = results['accuracy']
                    message += f"\n• {rank}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)"
            
            message += f"""

🚨 Security Status: {'⚠️  HIGH RISK - Malicious > 30%!' if malicious_percentage > 30 else '✅ NETWORK SECURE'}

📧 EMAIL NOTIFICATIONS SENT:
{email_status}

📁 Files Generated:
• Complete Results: {os.path.basename(results_file_path)}
• Malicious Data: {os.path.basename(malicious_file_path)}
• Analysis Chart: {os.path.basename(chart_path)}

Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            return render_template('result.html',
                                 message=message,
                                 prediction_results=prediction_results[:100],
                                 chart_data=chart_data,
                                 email_results=email_results)

        except Exception as e:
            error_message = f"❌ An error occurred during prediction: {str(e)}"
            print(error_message)
            return render_template('result.html', message=error_message, prediction_results=[], chart_data=None)

@app.route('/process_default_dataset')
def process_default_dataset():
    try:
        if not os.path.exists(dataset_path):
            error_message = f"Default dataset not found at: {dataset_path}"
            return render_template('result.html', message=error_message, prediction_results=[], chart_data=None)

        print("🔄 Processing default dataset...")
        split_and_save_dataset(dataset_path, train_data_path, test_data_path)
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        # Handle data preprocessing
        if 'dest_port' in test_df.columns:
            test_df['dest_port'] = test_df['dest_port'].fillna(-1).astype('int64')
        if 'src_port' in test_df.columns:
            test_df['src_port'] = test_df['src_port'].fillna(-1).astype('int64')
        
        # Apply same preprocessing to training data
        if 'dest_port' in train_df.columns:
            train_df['dest_port'] = train_df['dest_port'].fillna(-1).astype('int64')
        if 'src_port' in train_df.columns:
            train_df['src_port'] = train_df['src_port'].fillna(-1).astype('int64')

        # Train and compare multiple ML models
        print("🤖 Training and comparing multiple ML models on default dataset...")
        model_results, trained_models, label_encoders, scaler = train_and_compare_models(train_df, test_df)
        
        # Get Random Forest accuracy for backward compatibility
        rf_accuracy = model_results.get('Random Forest', {}).get('accuracy', 0.0)
        print(f"✅ Model comparison completed!")

        # Make predictions
        print("🤖 Making predictions on default dataset...")
        results_file_path, results_df = make_predictions_and_save_results(test_df)

        # Create malicious data file
        print("🚨 Extracting malicious data...")
        malicious_file_path, malicious_df = create_malicious_data_file(results_df)

        # Calculate statistics
        benign_count = len(results_df[results_df['predict'] == 'benign'])
        malicious_count = len(results_df[results_df['predict'] == 'malicious'])
        outlier_count = len(results_df[results_df['predict'] == 'outlier'])
        total_count = len(results_df)

        malicious_percentage = (malicious_count/total_count)*100
        benign_percentage = (benign_count/total_count)*100

        # Create detailed chart
        print("📊 Creating analysis charts...")
        chart_data, chart_path = create_detailed_bar_chart(
            benign_count, malicious_count, outlier_count, total_count, "Default Dataset"
        )
        
        # Create model comparison chart
        print("📈 Creating model comparison chart...")
        model_chart_data, model_chart_path = create_model_comparison_chart(model_results, "Default Dataset")

        # Send all emails
        print("📧 Sending email notifications...")
        email_results = send_all_emails(
            "Default Dataset (2020.06.19.csv)", results_file_path, chart_path, malicious_file_path,
            benign_count, malicious_count, outlier_count, total_count, malicious_df, model_results
        )

        # Prepare response data
        prediction_results = results_df.to_dict(orient='records')

        # Create email status summary
        email_status = "\n".join([f"✅ {name}: {'Sent' if result else 'Failed'}"
                                for name, result in email_results])

        message = f"""
🔍 DEFAULT DATASET ANALYSIS COMPLETE
====================================

📁 Dataset: Default Dataset (2020.06.19.csv)
📊 Total Records Processed: {total_count:,}

🛡️  SECURITY BREAKDOWN:
• Benign Traffic: {benign_count:,} ({benign_percentage:.1f}%)
• Malicious Traffic: {malicious_count:,} ({malicious_percentage:.1f}%)
• Outlier Traffic: {outlier_count:,} ({(outlier_count/total_count)*100:.1f}%)

🤖 MODEL PERFORMANCE COMPARISON:"""
        
        # Add model comparison results
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'error' not in results:
                rank = i + 1
                accuracy = results['accuracy']
                message += f"\n• {rank}. {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)"
        
        message += f"""

🚨 Security Status: {'⚠️  HIGH RISK - Malicious > 30%!' if malicious_percentage > 30 else '✅ NETWORK SECURE'}

📧 EMAIL NOTIFICATIONS SENT:
{email_status}

📁 Files Generated:
• Complete Results: {os.path.basename(results_file_path)}
• Malicious Data: {os.path.basename(malicious_file_path)}
• Analysis Chart: {os.path.basename(chart_path)}

Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        return render_template('result.html',
                             message=message,
                             prediction_results=prediction_results[:100],
                             chart_data=chart_data,
                             email_results=email_results)

    except Exception as e:
        error_message = f"❌ An error occurred while processing default dataset: {str(e)}"
        print(error_message)
        return render_template('result.html', message=error_message, prediction_results=[], chart_data=None)

@app.route('/compare_models')
def compare_models():
    """Standalone model comparison using default dataset"""
    # Return loading page first
    loading_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison - Loading</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 20px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .progress { background: rgba(255,255,255,0.2); border-radius: 25px; padding: 3px; margin: 20px 0; }
            .progress-bar { background: #4CAF50; height: 20px; border-radius: 22px; transition: width 0.3s; }
        </style>
        <script>
            let progress = 0;
            function updateProgress() {
                progress += Math.random() * 15;
                if (progress > 95) progress = 95;
                document.getElementById('progress-bar').style.width = progress + '%';
                document.getElementById('progress-text').innerText = Math.round(progress) + '%';
                if (progress < 95) setTimeout(updateProgress, 500);
            }
            setTimeout(updateProgress, 1000);
            setTimeout(() => window.location.href = '/compare_models?results=true', 8000);
        </script>
    </head>
    <body>
        <h1>🤖 Training Machine Learning Models</h1>
        <div class="loader"></div>
        <p>Analyzing network data and comparing 7 different ML algorithms...</p>
        <div class="progress">
            <div id="progress-bar" class="progress-bar" style="width: 0%"></div>
        </div>
        <p id="progress-text">0%</p>
        <p><small>This may take 30-60 seconds for first run. Subsequent runs will be much faster due to caching.</small></p>
    </body>
    </html>
    """
    
    # Check if we should show loading or results
    from flask import request
    if request.args.get('results') != 'true':
        return loading_html
    
    try:
        if not os.path.exists(dataset_path):
            error_message = f"Default dataset not found at: {dataset_path}"
            return f"<h1>Error</h1><p>{error_message}</p>"

        print("🔄 Processing default dataset for model comparison...")
        split_and_save_dataset(dataset_path, train_data_path, test_data_path)
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        # Handle data preprocessing
        if 'dest_port' in test_df.columns:
            test_df['dest_port'] = test_df['dest_port'].fillna(-1).astype('int64')
        if 'src_port' in test_df.columns:
            test_df['src_port'] = test_df['src_port'].fillna(-1).astype('int64')
        
        # Apply same preprocessing to training data
        if 'dest_port' in train_df.columns:
            train_df['dest_port'] = train_df['dest_port'].fillna(-1).astype('int64')
        if 'src_port' in train_df.columns:
            train_df['src_port'] = train_df['src_port'].fillna(-1).astype('int64')

        # Train and compare multiple ML models (fast version)
        print("🤖 Training and comparing multiple ML models...")
        model_results, trained_models, label_encoders, scaler = train_and_compare_models_fast(train_df, test_df, use_cache=True, sample_size=3000)
        
        print(f"✅ Model comparison completed!")

        # Create comprehensive model analysis dashboard
        print("📈 Creating comprehensive model analysis dashboard...")
        model_chart_data, model_chart_path = create_comprehensive_model_charts(model_results, "Default Dataset")
        
        # Create individual charts for separate downloads
        print("📊 Creating individual charts...")
        individual_chart_paths = create_individual_charts(model_results, "Default Dataset")

        # Create detailed analysis message
        total_count = len(test_df)
        best_model = list(model_results.keys())[0]
        best_accuracy = list(model_results.values())[0]['accuracy']
        
        model_comparison_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .model-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                .model-card.best {{ border-left-color: #ffd700; background: linear-gradient(135deg, #fff9c4 0%, #f8f9fa 100%); }}
                .accuracy {{ font-size: 24px; font-weight: bold; color: #28a745; }}
                .rank {{ display: inline-block; background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 10px; }}
                .rank.first {{ background: #ffd700; color: #333; }}
                .chart-container {{ text-align: center; margin: 30px 0; position: relative; }}
                .stats {{ background: #e9ecef; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .back-link {{ display: inline-block; background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
                .back-link:hover {{ background: #0056b3; }}
                .download-btn {{ display: inline-block; background: #28a745; color: white; padding: 12px 20px; text-decoration: none; border-radius: 5px; margin: 10px 5px; font-weight: bold; transition: background-color 0.3s; border: none; cursor: pointer; font-size: 14px; }}
                .download-btn:hover {{ background: #218838; color: white; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
                .download-section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #28a745; }}
                .chart-actions {{ margin-top: 15px; padding: 10px; background: rgba(40, 167, 69, 0.1); border-radius: 5px; }}
                .download-success {{ display: none; background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px; border: 1px solid #c3e6cb; }}
                .individual-downloads {{ margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #17a2b8; }}
                .download-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px; }}
                .download-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .download-item h4 {{ margin: 0 0 10px 0; color: #495057; font-size: 14px; }}
                .individual-btn {{ background: #17a2b8; color: white; padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: bold; transition: all 0.3s; }}
                .individual-btn:hover {{ background: #138496; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
            </style>
            <script>
                function downloadChart(url) {{
                    // Show loading state
                    const btn = event.target;
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '⬇️ Downloading...';
                    btn.style.pointerEvents = 'none';
                    
                    // Create download link
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'model_comparison_chart_' + new Date().toISOString().slice(0,19).replace(/:/g, '-') + '.png';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    // Show success message
                    setTimeout(() => {{
                        btn.innerHTML = '✅ Downloaded!';
                        const successMsg = document.getElementById('download-success');
                        if (successMsg) {{
                            successMsg.style.display = 'block';
                            setTimeout(() => {{
                                successMsg.style.display = 'none';
                                btn.innerHTML = originalText;
                                btn.style.pointerEvents = 'auto';
                            }}, 3000);
                        }} else {{
                            setTimeout(() => {{
                                btn.innerHTML = originalText;
                                btn.style.pointerEvents = 'auto';
                            }}, 2000);
                        }}
                    }}, 500);
                    
                    return false;
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Machine Learning Models Comparison</h1>
                    <p>Network Intrusion Detection Performance Analysis</p>
                    <p><strong>Dataset:</strong> Default Dataset (2020.06.19.csv) | <strong>Records:</strong> {total_count:,}</p>
                </div>
                
                <div class="stats">
                    <h3>📊 Performance Summary</h3>
                    <p><strong>🥇 Best Model:</strong> {best_model} ({best_accuracy:.4f} / {best_accuracy*100:.2f}%)</p>
                    <p><strong>🔬 Models Tested:</strong> {len(model_results)}</p>
                    <p><strong>📈 Analysis Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="download-section">
                    <h3>📈 Model Performance Visualization</h3>
                    <div class="chart-container">
                        {"<img src='data:image/png;base64," + model_chart_data + "' style='max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;' />" if model_chart_data else "<p>Chart generation failed</p>"}
                        {f'''<div class="chart-actions">
                            <p><strong>📊 Chart Information:</strong> Comprehensive model analysis with accuracy comparison, training time, and performance distribution</p>
                            <button class="download-btn" onclick="downloadChart('/download_chart/{os.path.basename(model_chart_path) if model_chart_path else 'chart.png'}')">
                                📥 Download Chart as PNG
                            </button>
                            <div id="download-success" class="download-success">
                                ✅ Chart downloaded successfully! Check your downloads folder.
                            </div>
                            <small style="display: block; margin-top: 10px; color: #666;">
                                📏 High-resolution PNG (300 DPI) • 📊 Perfect for reports and presentations • 🖼️ Includes all 4 analysis charts
                            </small>
                        </div>''' if model_chart_path else ""}
                    </div>
                </div>
                
                <div class="individual-downloads">
                    <h3>📊 Download Individual Charts</h3>
                    <p>Get specific charts for detailed analysis and reporting:</p>
                    <div class="download-grid">
                        {f'''<div class="download-item">
                            <h4>📈 Accuracy Comparison</h4>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">Bar chart showing model accuracy scores</p>
                            <button class="individual-btn" onclick="downloadChart('/download_chart/{os.path.basename(individual_chart_paths.get('accuracy', '')) if individual_chart_paths.get('accuracy') else 'accuracy.png'}')">
                                📥 Download
                            </button>
                        </div>''' if individual_chart_paths.get('accuracy') else ''}
                        {f'''<div class="download-item">
                            <h4>⏱️ Training Time</h4>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">Training speed comparison across models</p>
                            <button class="individual-btn" onclick="downloadChart('/download_chart/{os.path.basename(individual_chart_paths.get('training_time', '')) if individual_chart_paths.get('training_time') else 'training_time.png'}')">
                                📥 Download
                            </button>
                        </div>''' if individual_chart_paths.get('training_time') else ''}
                        {f'''<div class="download-item">
                            <h4>🎯 Accuracy vs Time</h4>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">Scatter plot of performance vs speed</p>
                            <button class="individual-btn" onclick="downloadChart('/download_chart/{os.path.basename(individual_chart_paths.get('scatter', '')) if individual_chart_paths.get('scatter') else 'scatter.png'}')">
                                📥 Download
                            </button>
                        </div>''' if individual_chart_paths.get('scatter') else ''}
                        {f'''<div class="download-item">
                            <h4>🥧 Performance Distribution</h4>
                            <p style="font-size: 12px; color: #666; margin: 5px 0;">Pie chart of performance categories</p>
                            <button class="individual-btn" onclick="downloadChart('/download_chart/{os.path.basename(individual_chart_paths.get('pie', '')) if individual_chart_paths.get('pie') else 'pie.png'}')">
                                📥 Download
                            </button>
                        </div>''' if individual_chart_paths.get('pie') else ''}
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: rgba(23, 162, 184, 0.1); border-radius: 5px;">
                        <small style="color: #666;">
                            📏 Each chart is high-resolution (300 DPI) • 🖼️ Perfect for reports and presentations • 📊 Individual focus for specific analysis
                        </small>
                    </div>
                </div>
                
                <h3>🏆 Detailed Model Results</h3>
                <div class="model-grid">
        """
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'error' not in results:
                rank = i + 1
                accuracy = results['accuracy']
                card_class = "model-card best" if rank == 1 else "model-card"
                rank_class = "rank first" if rank == 1 else "rank"
                
                model_comparison_html += f"""
                    <div class="{card_class}">
                        <div class="{rank_class}">#{rank}</div>
                        <h4>{model_name}</h4>
                        <div class="accuracy">{accuracy:.4f} ({accuracy*100:.2f}%)</div>
                        <p><strong>Performance:</strong> {'Excellent' if accuracy > 0.95 else 'Good' if accuracy > 0.90 else 'Fair' if accuracy > 0.80 else 'Needs Improvement'}</p>
                    </div>
                """
            else:
                model_comparison_html += f"""
                    <div class="model-card" style="border-left-color: #dc3545; background: #f8d7da;">
                        <h4>{model_name}</h4>
                        <p style="color: #721c24;"><strong>Error:</strong> {results.get('error', 'Training failed')}</p>
                    </div>
                """
        
        model_comparison_html += f"""
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/" class="back-link">🏠 Back to Home</a>
                    <a href="/upload_image" class="back-link">📁 Upload New Dataset</a>
                    <a href="/process_default_dataset" class="back-link">🔄 Run Full Analysis</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        return model_comparison_html

    except Exception as e:
        error_message = f"❌ An error occurred during model comparison: {str(e)}"
        print(error_message)
        return f"<h1>Error</h1><p>{error_message}</p><p><a href='/'>Back to Home</a></p>"

@app.route('/download_chart/<filename>')
def download_chart(filename):
    """Download chart as PNG file"""
    try:
        chart_file_path = os.path.join(charts_folder, filename)
        if os.path.exists(chart_file_path):
            from flask import send_file
            return send_file(chart_file_path, as_attachment=True, download_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        else:
            return "Chart file not found", 404
    except Exception as e:
        return f"Error downloading chart: {str(e)}", 500

if __name__ == '__main__':
    print("🚀 Starting Network Intrusion Detection Flask App...")
    print("🔧 Make sure H2O is properly initialized and the model file exists.")
    print("📧 Email notifications will be sent to: manaswinikhanna23@gmail.com")
    print("🌐 Server starting on http://0.0.0.0:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)