# Pneumonia Detection using Chest X-Rays

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Deployed on Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com)

##  Live Demo
Live demo application link: https://summative-ml-pipeline-g989.onrender.com

## Youtube Video
Youtube demo link: 

##  Project Overview
The pneumonia detection system uses a deep learning model that analyzes chest X-rays by identifying key visual patterns. It processes images through multiple convolutional layers to detect pneumonia indicators like lung consolidation, air bronchograms, and pleural effusion. The model was trained on thousands of labeled X-rays, learning to distinguish between normal lung tissue and pneumonia-related abnormalities. It outputs a confidence score for its prediction, with the web interface displaying the results along with visual heatmaps highlighting the areas that influenced the decision. The system includes features for continuous improvement, allowing retraining with new data to enhance accuracy over time.

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

## Model Tests

**Image 1:** Noticeable opacity in the right lung. Accurately highlights the pneumonia-affected region.

<img width="797" height="397" alt="image" src="https://github.com/user-attachments/assets/30d00d79-9d6c-487b-b05d-02a5b1acbb9b" />

**Image 2:** Normal X-ray: Clear lung fields with no abnormal opacities. Correctly identifies as normal with 96% confidence.

<img width="806" height="340" alt="image" src="https://github.com/user-attachments/assets/d5b56169-1158-4538-8dd0-6dc915ad7466" />

**Image 3:** Uploading New Data to retrain the model.

<img width="811" height="307" alt="image" src="https://github.com/user-attachments/assets/41334d79-9097-4fbd-b2e5-f5af1d151b74" /> 




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

##  Model Performance on Locust
The application demonstrates excellent scalability and reliability, making it suitable for production use. The zero failure rate indicates robust error handling, and the response times are well within acceptable limits for a medical imaging application.

<img width="889" height="387" alt="image" src="https://github.com/user-attachments/assets/c22e8255-c088-42e0-bae2-6884aa356e3e" />


<img width="731" height="391" alt="image" src="https://github.com/user-attachments/assets/9fde8699-b9ad-43ac-a1fc-78506f0370e6" />

<img width="892" height="431" alt="image" src="https://github.com/user-attachments/assets/49dbb123-c1f3-45f7-8094-812b5dddfba1" />



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

  Made by Denis Mitali | African Leadership University.
