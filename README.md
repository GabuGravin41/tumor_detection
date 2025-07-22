# Brain Tumor Detection Web Application

A Django-based web application for brain tumor detection and segmentation using deep learning models. The application provides both classification (tumor vs. healthy) and segmentation capabilities with interactive visualization.

## Features

- **Brain Tumor Classification**: Upload brain MRI images and get predictions with confidence scores
- **Grad-CAM Heatmaps**: Visualize model attention with color-coded heatmaps
- **Tumor Segmentation**: Segment tumor regions with interactive color options
- **Analysis History**: View and reanalyze previous uploads
- **PDF Reports**: Generate downloadable reports with images and analysis results
- **Interactive UI**: Modern, responsive interface with real-time feedback

## Prerequisites

- Python 3.8+
- Django 4.0+
- PyTorch
- TensorFlow/Keras
- OpenCV
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd brain_tumor_detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

### Download Model Weights

**Classification Model (PyTorch)**
1. Download the classification model from Kaggle:
   - Dataset: [Brain Tumor Classification Model](https://www.kaggle.com/datasets/your-dataset-url)
   - Download `Brain_Tumor_model.pt`
   - Place it in: `classification_brain_tumor_model/Brain_Tumor_model.pt`

**Segmentation Model (Keras)**
1. Download the segmentation model from Kaggle:
   - Dataset: [Brain Tumor Segmentation Model](https://www.kaggle.com/datasets/your-dataset-url)
   - Download the `.h5` model file
   - Place it in: `segmentation_braintumor_model/model.h5`

### Directory Structure
```
brain_tumor_detection/
├── classification_brain_tumor_model/
│   └── Brain_Tumor_model.pt          # Download from Kaggle
├── segmentation_braintumor_model/
│   └── model.h5                      # Download from Kaggle
├── tumor_detector/
│   ├── models.py                     # Model definitions
│   ├── views.py                      # Django views
│   └── urls.py                       # URL routing
├── templates/
│   └── tumor_detector/
│       └── home.html                 # Frontend template
├── media/                            # Uploaded images (auto-created)
├── requirements.txt
└── manage.py
```

## Configuration

1. **Database Setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

2. **Create Superuser (Optional)**
   ```bash
   python manage.py createsuperuser
   ```

3. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

4. **Access the Application**
   - Open browser and go to: `http://127.0.0.1:8000/`

## Usage

1. **Upload Image**: Click "Choose File" and select a brain MRI image
2. **Classification**: The system automatically runs classification and shows results
3. **Segmentation**: If tumor is detected, click "Segment Tumor" for detailed analysis
4. **Customize Colors**: Use the color picker to change segmentation overlay colors
5. **View History**: Check the sidebar for previous analyses
6. **Download Report**: Generate PDF reports with analysis results

## Model Integration Details

### Classification Model
- **Framework**: PyTorch
- **Architecture**: CNN_TUMOR (Custom CNN)
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Tumor/Healthy)
- **Features**: Grad-CAM heatmap generation

### Segmentation Model
- **Framework**: Keras/TensorFlow
- **Architecture**: U-Net
- **Input**: 256x256 grayscale images
- **Output**: Binary segmentation mask
- **Features**: Dice coefficient, IoU metrics

## API Endpoints

- `GET /` - Home page
- `POST /predict/` - Image classification
- `POST /segment/` - Tumor segmentation
- `GET /history/` - Analysis history
- `GET /report/<id>/` - Download PDF report
- `POST /delete/<id>/` - Delete analysis

## File Structure

```
├── brain_tumor_detection_web/     # Django project settings
├── tumor_detector/                # Main app
│   ├── models.py                  # ML models and Django models
│   ├── views.py                   # Request handlers
│   ├── urls.py                    # URL patterns
│   └── migrations/                # Database migrations
├── templates/                     # HTML templates
├── media/                         # User uploads (gitignored)
├── static/                        # Static files
└── requirements.txt               # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are in correct directories
   - Check file permissions
   - Verify model file integrity

2. **Memory Issues**
   - Reduce image size in settings
   - Use smaller batch sizes
   - Ensure sufficient RAM

3. **CUDA/GPU Issues**
   - Install appropriate PyTorch version
   - Check GPU drivers
   - Use CPU-only version if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Brain tumor datasets from Kaggle
- PyTorch and TensorFlow communities
- Django framework
- OpenCV for image processing
