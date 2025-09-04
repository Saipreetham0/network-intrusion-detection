# Network Intrusion Detection System (NIDS)

A sophisticated Flask-based web application for network intrusion detection using multiple machine learning algorithms including H2O framework. The system provides real-time network traffic analysis, threat detection, and comprehensive reporting with automated email notifications.

## Features

### Core Functionality
- **Web Interface**: Intuitive Flask-based web application with user authentication
- **File Upload**: Support for CSV dataset upload and analysis
- **Default Dataset**: Pre-loaded sample dataset (2020.06.19.csv) for immediate testing
- **Real-time Analysis**: Instant prediction results with detailed breakdowns

### Machine Learning Capabilities
- **Multiple ML Models**: Compares 7+ algorithms simultaneously:
  - Random Forest (Primary)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - Naive Bayes
  - K-Nearest Neighbors
  - Neural Network (MLP)
  - H2O Distributed Random Forest (DRF)
- **Model Caching**: Intelligent caching system for faster subsequent runs
- **Performance Comparison**: Detailed accuracy metrics and training time analysis
- **Auto-scaling**: Automatic data sampling for large datasets to optimize performance

### Security & Monitoring
- **Multi-class Detection**: Classifies traffic as benign, malicious, or outlier
- **Threat Level Assessment**: Automatic risk level calculation
- **Critical Alerts**: Automated alerts when malicious traffic exceeds 30%
- **Malicious Data Extraction**: Separate analysis of only suspicious traffic

### Reporting & Visualization
- **Visual Analytics**: Interactive charts and graphs for data visualization
- **Comprehensive Charts**: 
  - Security analysis with percentages
  - Model performance comparison
  - Training time analysis
  - Performance distribution
- **Email Notifications**: Multi-tiered automated email system:
  - Complete analysis results
  - Visual charts and summaries
  - Malicious data reports
  - Critical security alerts
- **Export Formats**: CSV files with detailed predictions and metadata

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd network-intrusion-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Java is installed (required for H2O):
```bash
java -version
```

## Usage

### Starting the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5001
```

3. Login with credentials:
   - Email: `manaswinikhanna23@gmail.com`
   - Password: `1234`

### Available Endpoints

- **`/`** - Login page
- **`/upload_image`** - File upload interface
- **`/predict`** - Process uploaded CSV file
- **`/process_default_dataset`** - Analyze the default dataset
- **`/compare_models`** - Compare ML model performance

### Analysis Options

1. **Upload Custom Dataset**: Upload your own CSV file for analysis
2. **Use Default Dataset**: Analyze the pre-loaded sample dataset
3. **Model Comparison**: Compare performance of different ML algorithms

### Understanding Results

The system provides comprehensive analysis including:
- **Traffic Classification**: Percentage breakdown of benign vs malicious traffic
- **Model Performance**: Accuracy scores for all tested algorithms
- **Visual Charts**: Interactive graphs showing analysis results
- **Risk Assessment**: Automated security status evaluation
- **Detailed Reports**: CSV exports with full prediction data

## Project Structure

```
network-intrusion-detection/
├── app.py                 # Main Flask application with ML pipeline
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates for web interface
│   ├── login.html        # User authentication
│   ├── upload.html       # File upload interface
│   └── result.html       # Analysis results display
├── saved_model/          # H2O pre-trained model files
│   └── DRF_model_*       # Distributed Random Forest models
├── model_cache/          # ML model results caching
├── results/              # Analysis output files
├── charts/               # Generated visualization files
├── malicious_data/       # Filtered malicious traffic data
├── data/                 # Processed training/testing datasets
├── 2020.06.19.csv       # Sample network traffic dataset
├── train.csv            # Training data split
├── test.csv             # Testing data split
├── *.joblib             # Scikit-learn model files
├── *.ipynb              # Jupyter notebooks for analysis
└── README.md            # This documentation
```

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: H2O.ai, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Email**: smtplib
- **Frontend**: HTML, CSS

## Model Information

### Primary Model
- **Algorithm**: Distributed Random Forest (DRF)
- **Framework**: H2O.ai
- **Classes**: benign, malicious, outlier
- **Training Split**: 80% training, 20% testing

### Model Comparison Suite
The system automatically compares multiple algorithms:

| Algorithm | Framework | Strengths | Use Case |
|-----------|-----------|-----------|----------|
| Random Forest | scikit-learn | High accuracy, handles missing data | Primary classifier |
| Logistic Regression | scikit-learn | Fast, interpretable | Linear relationships |
| SVM | scikit-learn | Effective with high-dimensional data | Complex patterns |
| Decision Tree | scikit-learn | Interpretable, fast training | Rule-based decisions |
| Naive Bayes | scikit-learn | Good with small datasets | Probabilistic classification |
| K-Nearest Neighbors | scikit-learn | Non-parametric, simple | Local pattern matching |
| Neural Network | scikit-learn | Complex pattern recognition | Non-linear relationships |
| H2O DRF | H2O.ai | Distributed processing | Large-scale data |

### Performance Metrics
- **Accuracy Score**: Primary evaluation metric
- **Training Time**: Efficiency measurement
- **Caching**: Results cached for faster subsequent runs
- **Auto-scaling**: Automatic data sampling for large datasets (default: 5000 samples)

## Email Configuration

The system provides a sophisticated multi-tier email notification system:

### Email Types
1. **Complete Analysis Results** - Full dataset analysis with all predictions
2. **Visual Charts & Summary** - High-resolution charts with percentage breakdowns
3. **Malicious Data Report** - Filtered dataset containing only suspicious traffic
4. **Critical Security Alert** - Automatic alert when malicious traffic exceeds 30%

### Configuration
Update email credentials in `app.py`:
```python
email_address = "your-email@gmail.com"
email_password = "your-app-password"  # Gmail App Password recommended
```

**Security Notes**:
- Use Gmail App Passwords instead of regular passwords
- In production, use environment variables:
  ```bash
  export EMAIL_ADDRESS="your-email@gmail.com"
  export EMAIL_PASSWORD="your-app-password"
  ```
- Never commit email credentials to version control

## Troubleshooting

### Common Issues

#### H2O Version Mismatch
If you encounter H2O version compatibility errors:
```bash
# Update H2O to latest version
pip install --upgrade h2o

# Or install specific version
pip install h2o==3.44.0.3
```

#### Memory Issues
For large datasets or memory constraints:
- Reduce `sample_size` parameter in `train_and_compare_models_fast()`
- Increase H2O memory allocation in `h2o.init(max_mem_size="4G")`
- Close other applications to free memory

#### Port Conflicts
If port 5001 is in use:
```python
# In app.py, change the port
app.run(debug=True, host='0.0.0.0', port=5002)
```

#### Email Issues
- Verify Gmail App Password is correct
- Check Gmail 2-factor authentication is enabled
- Ensure "Less secure app access" is disabled (use App Passwords instead)

#### File Upload Problems
- Ensure CSV files have proper headers
- Maximum file size may be limited by Flask configuration
- Check file permissions in upload directory

#### Model Loading Errors
- Verify saved_model directory contains H2O model files
- Check that H2O is properly initialized
- If model files are corrupted, the system will fall back to scikit-learn models

## Performance & Optimization

### Caching System
- **Model Results Caching**: Training results are cached based on data hash
- **Automatic Cache Invalidation**: Cache updates when dataset changes
- **Speed Improvement**: Subsequent runs up to 10x faster

### Data Processing
- **Smart Sampling**: Large datasets automatically sampled for faster processing
- **Parallel Processing**: Multi-threaded model training when possible
- **Memory Management**: Automatic memory optimization for different dataset sizes

### Scalability
- **Configurable Parameters**: Adjust sample sizes and model complexity
- **Distributed Computing**: H2O models support cluster deployment
- **Batch Processing**: Support for processing multiple files

## API Reference

### Main Endpoints

#### POST `/predict`
Process uploaded CSV file for intrusion detection.

**Parameters:**
- `file`: CSV file containing network traffic data

**Response:**
- Analysis results with predictions, charts, and email notifications

#### GET `/process_default_dataset`
Analyze the default sample dataset.

**Response:**
- Complete analysis results using pre-loaded dataset

#### GET `/compare_models`
Compare performance of different ML algorithms.

**Response:**
- Interactive dashboard showing model comparison results

## Dataset Format

### Required Columns
The system expects CSV files with network traffic features. Common columns include:
- `src_ip`, `dest_ip`: IP addresses
- `src_port`, `dest_port`: Port numbers
- `protocol`: Network protocol
- `bytes`, `packets`: Traffic volume metrics
- `label`/`class`: Ground truth labels (for training data)

### Data Preprocessing
- **Missing Values**: Automatically handled with imputation
- **Categorical Encoding**: Label encoding for non-numeric features
- **Feature Scaling**: StandardScaler applied to relevant algorithms
- **Data Validation**: Automatic data quality checks

## Security Considerations

### Authentication
- Simple login system (replace with robust authentication in production)
- Session management for user access control

### Data Privacy
- Local processing only (no data sent to external services)
- Temporary file cleanup after processing
- Email attachments compressed for security

### Production Recommendations
1. **Implement proper authentication** (OAuth, LDAP, etc.)
2. **Use HTTPS** with SSL certificates
3. **Set up proper logging** and monitoring
4. **Use environment variables** for sensitive configuration
5. **Implement rate limiting** to prevent abuse
6. **Regular security updates** for all dependencies

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv nids_env
   source nids_env/bin/activate  # On Windows: nids_env\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Development tools
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. Make your changes and test thoroughly
6. Submit a pull request with detailed description

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include unit tests for new features
- Update documentation for API changes

## License

This project is for educational and research purposes. See LICENSE file for details.

## Acknowledgments

- H2O.ai for the distributed machine learning platform
- scikit-learn for classical ML algorithms
- Flask community for the web framework
- Network security research community for datasets and methodologies

## Contact

- **Repository**: [GitHub Issues](https://github.com/Saipreetham0/network-intrusion-detection/issues)
- **Email**: For direct contact regarding collaboration or research
- **Documentation**: This README and inline code documentation

---

**Disclaimer**: This tool is intended for educational and research purposes. Always ensure you have proper authorization before analyzing network traffic in any environment.