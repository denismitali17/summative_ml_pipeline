// Main JavaScript for Pneumonia Prediction

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Account for fixed header
                    behavior: 'smooth'
                });
            }
        });
    });

    // Initialize charts (if Chart.js is included)
    initializeCharts();

    // Add animation to elements with 'animate-on-scroll' class
    setupScrollAnimations();

    // Initialize file upload previews
    setupFileUploads();
});

/**
 * Initialize charts using Chart.js
 */
function initializeCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') return;

    // Class Distribution Chart (Pie Chart)
    const classCtx = document.getElementById('classDistributionChart');
    if (classCtx) {
        // Mock data - replace with actual data from your API
        const classData = {
            labels: ['Lion', 'Leopard', 'Elephant', 'Buffalo', 'Rhino'],
            datasets: [{
                data: [25, 20, 30, 15, 10],
                backgroundColor: [
                    '#3498db',
                    '#e74c3c',
                    '#2ecc71',
                    '#f39c12',
                    '#9b59b6'
                ],
                borderWidth: 1
            }]
        };

        new Chart(classCtx, {
            type: 'doughnut',
            data: classData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                    },
                    title: {
                        display: true,
                        text: 'Class Distribution',
                        font: {
                            size: 16
                        }
                    }
                },
                cutout: '70%',
                animation: {
                    animateScale: true,
                    animateRotate: true
                }
            }
        });
    }

    // Training History Chart (Line Chart)
    const historyCtx = document.getElementById('trainingHistoryChart');
    if (historyCtx) {
        // Mock data - replace with actual data from your API
        const historyData = {
            labels: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5', 'Epoch 6', 'Epoch 7', 'Epoch 8', 'Epoch 9', 'Epoch 10'],
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Accuracy',
                    data: [0.63, 0.70, 0.76, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        };

        new Chart(historyCtx, {
            type: 'line',
            data: historyData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training History',
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.5,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
}

/**
 * Setup scroll animations for elements with 'animate-on-scroll' class
 */
function setupScrollAnimations() {
    const animateElements = document.querySelectorAll('.animate-on-scroll');
    
    if (!('IntersectionObserver' in window)) {
        // If IntersectionObserver is not supported, add the 'animated' class immediately
        animateElements.forEach(el => el.classList.add('animated'));
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    animateElements.forEach(el => observer.observe(el));
}

/**
 * Setup file upload functionality
 */
function setupFileUploads() {
    // Image upload preview
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const noPreview = document.getElementById('noPreview');
    
    if (imageUpload && preview) {
        imageUpload.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    preview.src = event.target.result;
                    preview.classList.remove('d-none');
                    if (noPreview) noPreview.classList.add('d-none');
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Custom file input styling
    const customFileInputs = document.querySelectorAll('.custom-file-input');
    customFileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : 'Choose file';
            const label = this.nextElementSibling;
            if (label && label.classList.contains('custom-file-label')) {
                label.textContent = fileName;
            }
        });
    });
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of notification (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    // Check if toast container exists, if not create it
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.position = 'fixed';
        toastContainer.style.top = '20px';
        toastContainer.style.right = '20px';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }

    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast show align-items-center text-white bg-${type} border-0`;
    toast.role = 'alert';
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    const toastBody = document.createElement('div');
    toastBody.className = 'd-flex';
    
    const toastMessage = document.createElement('div');
    toastMessage.className = 'toast-body';
    toastMessage.textContent = message;
    
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close btn-close-white me-2 m-auto';
    closeButton.setAttribute('data-bs-dismiss', 'toast');
    closeButton.setAttribute('aria-label', 'Close');
    
    closeButton.addEventListener('click', function() {
        toast.remove();
    });
    
    toastBody.appendChild(toastMessage);
    toastBody.appendChild(closeButton);
    toast.appendChild(toastBody);
    
    // Add to container and auto-remove after 5 seconds
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

/**
 * Format bytes to human-readable format
 * @param {number} bytes - The size in bytes
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted file size
 */
function formatFileSize(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    
        const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Main application state
const appState = {
    currentTab: 'predict',
    modelStatus: {
        loaded: false,
        version: null,
        numClasses: 0,
        classNames: []
    },
    trainingStatus: {
        status: 'idle',
        progress: 0,
        message: '',
        startTime: null,
        endTime: null
    },
    metrics: {
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1_score: 0,
        lastTrained: null
    }
};

// DOM Elements
const elements = {
    // Tabs
    predictTab: document.getElementById('predict-tab'),
    retrainTab: document.getElementById('retrain-tab'),
    metricsTab: document.getElementById('metrics-tab'),
    aboutTab: document.getElementById('about-tab'),
    
    // Tab content
    predictContent: document.getElementById('predict-content'),
    retrainContent: document.getElementById('retrain-content'),
    metricsContent: document.getElementById('metrics-content'),
    aboutContent: document.getElementById('about-content'),
    
    // Prediction form
    predictForm: document.getElementById('predict-form'),
    fileInput: document.getElementById('file-input'),
    previewImage: document.getElementById('preview-image'),
    predictButton: document.getElementById('predict-button'),
    predictionResult: document.getElementById('prediction-result'),
    confidenceBar: document.getElementById('confidence-bar'),
    confidenceText: document.getElementById('confidence-text'),
    predictedClass: document.getElementById('predicted-class'),
    
    // Retrain form
    retrainForm: document.getElementById('retrain-form'),
    trainingDataInput: document.getElementById('training-data'),
    epochsInput: document.getElementById('epochs'),
    learningRateInput: document.getElementById('learning-rate'),
    startTrainingButton: document.getElementById('start-training'),
    trainingProgress: document.getElementById('training-progress'),
    trainingStatus: document.getElementById('training-status'),
    trainingProgressBar: document.getElementById('training-progress-bar'),
    
    // Metrics
    modelVersion: document.getElementById('model-version'),
    modelAccuracy: document.getElementById('model-accuracy'),
    modelPrecision: document.getElementById('model-precision'),
    modelRecall: document.getElementById('model-recall'),
    modelF1Score: document.getElementById('model-f1-score'),
    lastTrained: document.getElementById('last-trained'),
    classDistributionChart: document.getElementById('class-distribution-chart'),
    trainingHistoryChart: document.getElementById('training-history-chart')
};

// Initialize the application
function init() {
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    updateModelStatus();
    updateMetrics();
    
    // Show the default tab
    showTab('predict');
}

// Set up event listeners
function setupEventListeners() {
    // Tab navigation
    elements.predictTab.addEventListener('click', () => showTab('predict'));
    elements.retrainTab.addEventListener('click', () => showTab('retrain'));
    elements.metricsTab.addEventListener('click', () => showTab('metrics'));
    elements.aboutTab.addEventListener('click', () => showTab('about'));
    
    // File input preview
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Form submissions
    elements.predictForm.addEventListener('submit', handlePrediction);
    elements.retrainForm.addEventListener('submit', handleRetraining);
    
    // Start with predict tab active
    elements.predictTab.classList.add('active');
    elements.predictContent.classList.add('active');
}

// Show the selected tab
function showTab(tabName) {
    // Hide all tab content
    elements.predictContent.classList.remove('active');
    elements.retrainContent.classList.remove('active');
    elements.metricsContent.classList.remove('active');
    elements.aboutContent.classList.remove('active');
    
    // Remove active class from all tabs
    elements.predictTab.classList.remove('active');
    elements.retrainTab.classList.remove('active');
    elements.metricsTab.classList.remove('active');
    elements.aboutTab.classList.remove('active');
    
    // Show the selected tab content and set the tab as active
    switch (tabName) {
        case 'predict':
            elements.predictContent.classList.add('active');
            elements.predictTab.classList.add('active');
            break;
        case 'retrain':
            elements.retrainContent.classList.add('active');
            elements.retrainTab.classList.add('active');
            break;
        case 'metrics':
            elements.metricsContent.classList.add('active');
            elements.metricsTab.classList.add('active');
            updateMetrics();
            break;
        case 'about':
            elements.aboutContent.classList.add('active');
            elements.aboutTab.classList.add('active');
            break;
    }
}

// Handle file selection for prediction
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            elements.previewImage.src = e.target.result;
            elements.previewImage.style.display = 'block';
            elements.predictButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

// Handle prediction form submission
async function handlePrediction(event) {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('file', elements.fileInput.files[0]);
    
    try {
        // Show loading state
        elements.predictButton.disabled = true;
        elements.predictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';
        
        // Send prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update UI with prediction result
        displayPredictionResult(data.prediction);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Error: ' + error.message, 'error');
    } finally {
        // Reset button state
        elements.predictButton.disabled = false;
        elements.predictButton.innerHTML = 'Predict';
    }
}

// Display prediction result
function displayPredictionResult(prediction) {
    // Update prediction text
    elements.predictedClass.textContent = prediction.class;
    elements.confidenceText.textContent = `${(prediction.confidence * 100).toFixed(2)}%`;
    
    // Update progress bar
    const confidencePercent = Math.round(prediction.confidence * 100);
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    elements.confidenceBar.setAttribute('aria-valuenow', confidencePercent);
    
    // Show the result section
    elements.predictionResult.style.display = 'block';
    
    // Scroll to the result
    elements.predictionResult.scrollIntoView({ behavior: 'smooth' });
}

// Handle retraining form submission
async function handleRetraining(event) {
    event.preventDefault();
    
    if (elements.trainingDataInput.files.length === 0) {
        showToast('Please select a zip file containing training data', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', elements.trainingDataInput.files[0]);
    formData.append('epochs', elements.epochsInput.value);
    formData.append('learning_rate', elements.learningRateInput.value);
    
    try {
        // Show loading state
        elements.startTrainingButton.disabled = true;
        elements.startTrainingButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        
        // Show training status
        elements.trainingProgress.style.display = 'block';
        updateTrainingStatus('Starting training...', 0);
        
        // Start retraining
        const response = await fetch('/api/retrain', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Start polling for training status
        pollTrainingStatus();
        
    } catch (error) {
        console.error('Retraining error:', error);
        showToast('Error: ' + error.message, 'error');
        updateTrainingStatus('Training failed: ' + error.message, 0, 'error');
        elements.startTrainingButton.disabled = false;
        elements.startTrainingButton.innerHTML = 'Start Training';
    }
}

// Poll for training status
async function pollTrainingStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        // Update UI with current status
        const status = data.training_status;
        appState.trainingStatus = status;
        
        // Update progress bar and status
        if (status.status === 'training' || status.status === 'completed') {
            const progress = status.progress || 0;
            updateTrainingStatus(status.message, progress, status.status);
            
            // Continue polling if still training
            if (status.status === 'training') {
                setTimeout(pollTrainingStatus, 2000);
            } else {
                // Training completed
                elements.startTrainingButton.disabled = false;
                elements.startTrainingButton.innerHTML = 'Start Training';
                
                // Update model status and metrics
                updateModelStatus();
                updateMetrics();
                
                if (status.status === 'completed') {
                    showToast('Model retraining completed successfully!', 'success');
                }
            }
        }
    } catch (error) {
        console.error('Error checking training status:', error);
        updateTrainingStatus('Error checking training status', 0, 'error');
        elements.startTrainingButton.disabled = false;
        elements.startTrainingButton.innerHTML = 'Start Training';
    }
}

// Update training status UI
function updateTrainingStatus(message, progress, status = 'info') {
    elements.trainingStatus.textContent = message;
    elements.trainingProgressBar.style.width = `${progress}%`;
    elements.trainingProgressBar.setAttribute('aria-valuenow', progress);
    elements.trainingProgressBar.textContent = `${Math.round(progress)}%`;
    
    // Update progress bar color based on status
    const progressBar = elements.trainingProgressBar;
    progressBar.className = 'progress-bar progress-bar-striped';
    
    if (status === 'error') {
        progressBar.classList.add('bg-danger');
    } else if (status === 'completed') {
        progressBar.classList.add('bg-success');
    } else if (status === 'training') {
        progressBar.classList.add('progress-bar-animated');
    }
}

// Update model status from the server
async function updateModelStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        appState.modelStatus = {
            loaded: data.model_status.loaded,
            version: data.model_status.version,
            numClasses: data.model_status.num_classes,
            classNames: data.model_status.class_names || []
        };
        
        appState.metrics = {
            accuracy: data.metrics.accuracy,
            precision: data.metrics.precision,
            recall: data.metrics.recall,
            f1_score: data.metrics.f1_score,
            lastTrained: data.metrics.last_trained
        };
        
        // Update UI
        updateMetricsUI();
        
    } catch (error) {
        console.error('Error updating model status:', error);
    }
}

// Update metrics UI
function updateMetrics() {
    if (elements.metricsContent.classList.contains('active')) {
        updateModelStatus();
    }
}

// Update metrics UI elements
function updateMetricsUI() {
    // Update model info
    elements.modelVersion.textContent = appState.modelStatus.version || 'N/A';
    elements.modelAccuracy.textContent = (appState.metrics.accuracy * 100).toFixed(2) + '%';
    elements.modelPrecision.textContent = (appState.metrics.precision * 100).toFixed(2) + '%';
    elements.modelRecall.textContent = (appState.metrics.recall * 100).toFixed(2) + '%';
    elements.modelF1Score.textContent = appState.metrics.f1_score.toFixed(4);
    
    // Format last trained time
    if (appState.metrics.lastTrained) {
        const lastTrained = new Date(appState.metrics.lastTrained * 1000);
        elements.lastTrained.textContent = lastTrained.toLocaleString();
    } else {
        elements.lastTrained.textContent = 'Never';
    }
    
    // Update charts
    updateCharts();
}

// Update charts
function updateCharts() {
    // Class distribution chart (mock data for now)
    const classDistributionCtx = elements.classDistributionChart.getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.classDistributionChart) {
        window.classDistributionChart.destroy();
    }
    
    // Create class distribution chart
    if (appState.modelStatus.classNames.length > 0) {
        const classCounts = appState.modelStatus.classNames.map(() => 
            Math.floor(Math.random() * 100) + 50
        );
        
        window.classDistributionChart = new Chart(classDistributionCtx, {
            type: 'bar',
            data: {
                labels: appState.modelStatus.classNames,
                datasets: [{
                    label: 'Number of Samples',
                    data: classCounts,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Samples'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Class'
                        }
                    }
                }
            }
        });
    }
    
    // Training history chart (mock data for now)
    const trainingHistoryCtx = elements.trainingHistoryChart.getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.trainingHistoryChart) {
        window.trainingHistoryChart.destroy();
    }
    
    // Create training history chart
    const epochs = Array.from({ length: 10 }, (_, i) => i + 1);
    const accuracy = epochs.map(e => Math.min(0.95, 0.7 + (Math.random() * 0.3 * e / 10)));
    const valAccuracy = accuracy.map(a => Math.max(0.6, a - (Math.random() * 0.1)));
    const loss = epochs.map(e => Math.max(0.1, 1 - (Math.random() * 0.9 * e / 10)));
    const valLoss = loss.map(l => l + (Math.random() * 0.2));
    
    window.trainingHistoryChart = new Chart(trainingHistoryCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: accuracy,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Validation Accuracy',
                    data: valAccuracy,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Training Loss',
                    data: loss,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    yAxisID: 'y1'
                },
                {
                    label: 'Validation Loss',
                    data: valLoss,
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Accuracy'
                    },
                    min: 0,
                    max: 1
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    min: 0,
                    max: Math.max(...loss, ...valLoss) * 1.1,
                    grid: {
                        drawOnChartArea: false
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                }
            }
        }
    });
}

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add to container
    const toastContainer = document.getElementById('toast-container');
    toastContainer.appendChild(toast);
    
    // Initialize and show the toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
    
    // Remove the toast after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
    
    bsToast.show();
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', init);

// Export functions to make them available globally
window.app = {
    showToast,
    formatFileSize
};
