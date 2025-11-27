# Pneumonia Detection using Chest X-Rays

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Deployed on Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com)

##  Live Demo
Access the live application: [Pneumonia detection](https://summative-ml-pipeline-g989.onrender.com/)

##  Project Overview
An end-to-end machine learning pipeline for detecting pneumonia from chest X-ray images, featuring model training, evaluation, and a web interface for predictions and model retraining.

##  Features
- Real-time X-ray analysis
- Model retraining with new data
- Performance monitoring dashboard
- RESTful API for integration
- Responsive web interface

##  Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: PyTorch, scikit-learn
- **Deployment**: Render
- **Testing**: Locust

##  Model Performance
| Metric        | Score   |
|---------------|---------|
| Accuracy      | 95.75%  |
| Precision     | 95.84%  |
| Recall        | 95.75%  |
| F1-Score      | 95.78%  |
| AUC-ROC       | 0.9915  |

##  Getting Started

### Prerequisites
- Python 3.9+
- Git
- Render account (for deployment)

### Local Development
```bash
# Clone repository
git clone https://github.com/denismitali17/summative_ml_pipeline
cd pneumonia-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

##  API Endpoints
- `POST /api/predict` - Make prediction on X-ray image
- `POST /api/retrain` - Retrain model with new data
- `GET /api/status` - Check model status
- `GET /api/classes` - Get available classes

##  Deployment on Render

### 1. Create a new Web Service
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New" → "Web Service"
3. Connect your GitHub repository

### 2. Configure Web Service
- **Name**: pneumonia-detection
- **Region**: Choose closest to your users
- **Branch**: main
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

### 3. Environment Variables
Add these environment variables:
```
PYTHON_VERSION=3.9.16
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

### 4. Deploy
Click "Create Web Service" to deploy

##  Usage

### Making Predictions
1. Go to the web interface
2. Upload a chest X-ray image
3. View the prediction results

### Retraining the Model
1. Prepare a ZIP file with training data:
   ```
   new_data/
   ├── NORMAL/
   │   └── normal1.jpeg
   └── PNEUMONIA/
       └── pneumonia1.jpeg
   ```
2. Upload via the "Retrain Model" section

##  Load Testing
```bash
# Install Locust
pip install locust

# Run load test
locust -f locustfile.py
```

### Load Test Results
| Users | Avg Response Time | Requests/sec | Failures |
|-------|-------------------|--------------|----------|
| 100   | 450ms             | 45           | 0%       |
| 500   | 1.1s              | 120          | 0%       |
| 1000  | 2.3s              | 200          | 2%       |

##  Project Structure
```
summative_ml_pipeline/
├── app/                    # Web application
│   ├── static/             # Static files
│   └── templates/          # HTML templates
├── data/                   # Dataset
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── data_loader.py      # Data processing
│   ├── model.py            # Model architecture
│   ├── predict.py          # Prediction logic
│   └── train.py            # Training pipeline
├── tests/                  # Unit tests
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Icons: [Font Awesome](https://fontawesome.com/)
- UI: [Bootstrap 5](https://getbootstrap.com/)

##  Notes
- This tool is for educational purposes only
- Not for clinical diagnosis
- Always consult a healthcare professional

##  Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  Made [Denis Mitali] | African Leadership University.
</div>
```
