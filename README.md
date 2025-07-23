# Healthcare AI - Chest X-ray Pneumonia Detection

A production-ready deep learning system for automated pneumonia detection in chest X-ray images using state-of-the-art computer vision techniques.

## 🎯 Overview

This project implements a comprehensive healthcare AI solution that can classify chest X-ray images into three categories:
- **Normal**: Healthy chest X-rays with no signs of pneumonia
- **Bacterial Pneumonia**: X-rays showing signs of bacterial pneumonia
- **Viral Pneumonia**: X-rays showing signs of viral pneumonia

The system uses a ResNet-based architecture with transfer learning, comprehensive data augmentation, and robust evaluation metrics to achieve high accuracy in medical image classification.

## 🚀 Features

- **State-of-the-art Model Architecture**: ResNet50 with custom classification head
- **Comprehensive Data Pipeline**: Support for DICOM and standard image formats
- **Advanced Augmentation**: Medical image-specific data augmentation techniques
- **Production-Ready Training**: Mixed precision, early stopping, learning rate scheduling
- **Extensive Evaluation**: ROC curves, confusion matrices, classification reports
- **Real-time Inference**: Batch and single image prediction with confidence scoring
- **Web Interface**: Streamlit-based UI for easy interaction
- **Docker Support**: Containerized deployment with CPU/GPU support
- **Comprehensive Testing**: Unit tests and integration tests

## 📊 Data Strategy

### Supported Data Sources
- **NIH Chest X-ray Dataset**: Large-scale chest X-ray dataset
- **RSNA Pneumonia Detection Challenge**: Competition dataset
- **Custom Medical Datasets**: Support for DICOM and standard formats

### Data Specifications
```json
{
  "image_format": ["PNG", "JPEG", "DICOM"],
  "resolution": "512x512",
  "channels": 1,
  "augmentation": [
    "rotation (±15°)",
    "horizontal_flip",
    "brightness/contrast adjustment",
    "gaussian_noise",
    "blur"
  ],
  "data_split": {
    "train": "70%",
    "validation": "15%",
    "test": "15%"
  }
}
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Docker (optional)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-org/healthcare-ai-pneumonia.git
cd healthcare-ai-pneumonia

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Docker Installation
```bash
# Build the image
docker build -t healthcare-ai .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data healthcare-ai

# Run with CPU only
docker run -v $(pwd)/data:/app/data healthcare-ai
```

## 📁 Project Structure

```
healthcare-ai-pneumonia/
├── config.yaml              # Configuration file
├── requirements.txt          # Python dependencies
├── data_loader.py           # Data loading and preprocessing
├── model.py                 # Model architecture and management
├── train.py                 # Training pipeline
├── inference.py             # Inference engine
├── evaluate.py              # Evaluation metrics and visualization
├── test_model.py            # Unit tests
├── streamlit_app.py         # Web interface
├── Dockerfile               # Container configuration
├── README.md                # This file
├── data/                    # Data directory
│   ├── train/
│   │   ├── Normal/
│   │   ├── Bacterial_Pneumonia/
│   │   └── Viral_Pneumonia/
│   ├── val/
│   └── test/
├── models/                  # Saved models
├── logs/                    # Training logs
├── checkpoints/             # Model checkpoints
└── evaluation_results/      # Evaluation outputs
```

## 🎓 Usage

### 1. Data Preparation

Organize your data in the following structure:
```
data/
├── train/
│   ├── Normal/
│   ├── Bacterial_Pneumonia/
│   └── Viral_Pneumonia/
├── val/
│   ├── Normal/
│   ├── Bacterial_Pneumonia/
│   └── Viral_Pneumonia/
└── test/
    ├── Normal/
    ├── Bacterial_Pneumonia/
    └── Viral_Pneumonia/
```

### 2. Training

```bash
# Basic training
python train.py

# Training with custom config
python train.py --config custom_config.yaml

# Training with experiment tracking
python train.py --experiment "pneumonia_resnet50_v1"

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Evaluation

```bash
# Evaluate on test set
python evaluate.py --model checkpoints/best_model.pth

# Evaluate on validation set
python evaluate.py --model checkpoints/best_model.pth --split val

# Save results to custom directory
python evaluate.py --model checkpoints/best_model.pth --output custom_eval_results/
```

### 4. Inference

```bash
# Single image prediction
python inference.py --model checkpoints/best_model.pth --input path/to/image.png

# Batch prediction on directory
python inference.py --model checkpoints/best_model.pth --input path/to/images/ --output results.csv

# Prediction with visualization
python inference.py --model checkpoints/best_model.pth --input image.png --visualize

# Generate detailed report
python inference.py --model checkpoints/best_model.pth --input path/to/images/ --report
```

### 5. Web Interface

```bash
# Launch Streamlit app
streamlit run streamlit_app.py

# Access at http://localhost:8501
```

## ⚙️ Configuration

The system is highly configurable through `config.yaml`:

```yaml
# Model configuration
model:
  architecture: "resnet50"
  pretrained: true
  freeze_backbone: false
  dropout_rate: 0.5

# Data configuration
data:
  data_dir: "data"
  image_size: [512, 512]
  num_classes: 3
  class_names: ["Normal", "Bacterial_Pneumonia", "Viral_Pneumonia"]
  augmentation:
    enabled: true
    rotation_range: 15
    horizontal_flip: true

# Training configuration
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping:
    enabled: true
    patience: 10

# Hardware configuration
hardware:
  device: "auto"  # auto, cpu, cuda
  mixed_precision: true
  num_workers: 4
```

## 📈 Model Performance

### Expected Results
- **Accuracy**: 85-92%
- **F1-Score (Macro)**: 0.83-0.90
- **AUC (Weighted)**: 0.88-0.95

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves for each class
- Confusion matrices
- Classification reports
- Per-class performance analysis

## 🧪 Testing

```bash
# Run all tests
python -m pytest test_model.py -v

# Run specific test class
python -m pytest test_model.py::TestMedicalImageClassifier -v

# Run with coverage
python -m pytest test_model.py --cov=. --cov-report=html
```

## 🚀 Deployment

### Local Deployment
```bash
# Train model
python train.py --experiment production_v1

# Start inference server
python inference.py --model checkpoints/best_model.pth --input data/test/

# Launch web interface
streamlit run streamlit_app.py
```

### Docker Deployment
```bash
# Build and run
docker build -t healthcare-ai .
docker run -p 8501:8501 -v $(pwd)/data:/app/data healthcare-ai

# With GPU support
docker run --gpus all -p 8501:8501 -v $(pwd)/data:/app/data healthcare-ai
```

### Cloud Deployment
The system supports deployment on:
- **AWS**: EC2 with GPU instances, S3 for data storage
- **Google Cloud**: Compute Engine, Cloud Storage
- **Azure**: Virtual Machines, Blob Storage

## 🔧 Development

### Code Quality
```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
pylint *.py
```

### Adding New Features
1. Update model architecture in `model.py`
2. Modify data pipeline in `data_loader.py`
3. Add evaluation metrics in `evaluate.py`
4. Update configuration in `config.yaml`
5. Add tests in `test_model.py`

## 📊 Monitoring and Logging

### Training Monitoring
- **TensorBoard**: Real-time training metrics
- **Weights & Biases**: Experiment tracking (optional)
- **Console Logging**: Detailed progress information

### Production Monitoring
- Model performance metrics
- Inference latency tracking
- Error rate monitoring
- Resource utilization

## 🔒 Healthcare Compliance

### Data Privacy
- HIPAA compliance considerations
- Data anonymization support
- Secure data handling practices

### Model Validation
- Cross-validation strategies
- Clinical validation protocols
- Performance monitoring in production

## 📚 References

1. **Medical Imaging**: Chest X-ray analysis techniques
2. **Deep Learning**: ResNet architecture and transfer learning
3. **Healthcare AI**: Medical image classification best practices
4. **Data Augmentation**: Medical image-specific augmentation strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
1. **CUDA out of memory**: Reduce batch size in config
2. **Data loading errors**: Check file paths and formats
3. **Model convergence**: Adjust learning rate and augmentation

### Getting Help
- Check the [Issues](https://github.com/your-org/healthcare-ai-pneumonia/issues) page
- Review the documentation
- Contact the development team

## 🎯 Roadmap

### Version 2.0
- [ ] Multi-modal input support (clinical data + images)
- [ ] Federated learning capabilities
- [ ] Real-time streaming inference
- [ ] Advanced visualization tools
- [ ] Integration with PACS systems

### Version 1.1
- [ ] Additional model architectures (EfficientNet, Vision Transformer)
- [ ] Automated hyperparameter tuning
- [ ] Model explainability features (Grad-CAM)
- [ ] API endpoints for integration

## 🏆 Acknowledgments

- Medical imaging datasets and research community
- PyTorch and deep learning frameworks
- Healthcare AI research and best practices
- Open source contributors and maintainers

---

**⚠️ Medical Disclaimer**: This software is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult with qualified healthcare professionals for medical decisions.
