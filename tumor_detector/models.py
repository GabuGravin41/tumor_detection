import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from django.db import models
from django.utils import timezone
import uuid
import os
import cv2
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf

# --- Segmentation Model (Keras UNet) ---
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras import backend as keras_K

# Custom metrics and loss for segmentation

def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = keras_K.flatten(y_true)
    y_pred_flatten = keras_K.flatten(y_pred)
    intersection = keras_K.sum(y_true_flatten * y_pred_flatten)
    union = keras_K.sum(y_true_flatten) + keras_K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

def iou_coef(y_true, y_pred, smooth=100):
    intersection = keras_K.sum(y_true * y_pred)
    sum_ = keras_K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return iou

class SegmentationBrainTumorModel:
    def __init__(self, model_path):
        self.model = keras_load_model(
            model_path,
            custom_objects={
                'dice_coef': dice_coef,
                'dice_loss': dice_loss,
                'iou_coef': iou_coef
            }
        )
        self.input_size = (256, 256)

    def preprocess_image(self, image_path):
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.input_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_mask(self, image_path):
        img = self.preprocess_image(image_path)
        pred = self.model.predict(img)[0, :, :, 0]
        mask = (pred > 0.5).astype('uint8')
        return mask

    def overlay_mask(self, image_path, mask, color_map='green'):
        import cv2
        import numpy as np
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.input_size)
        color_mask = np.zeros_like(img)
        if color_map == 'green':
            color_mask[:, :, 1] = mask * 255  # Green
        elif color_map == 'red':
            color_mask[:, :, 2] = mask * 255  # Red
        elif color_map == 'cyan':
            color_mask[:, :, 0] = mask * 255  # Blue
            color_mask[:, :, 1] = mask * 255  # Green
        elif color_map == 'magenta':
            color_mask[:, :, 0] = mask * 255  # Blue
            color_mask[:, :, 2] = mask * 255  # Red
        elif color_map == 'yellow':
            color_mask[:, :, 1] = mask * 255  # Green
            color_mask[:, :, 2] = mask * 255  # Red
        else:
            color_mask[:, :, 1] = mask * 255  # Default to green
        overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        return overlay


def findConv2dOutShape(H_in, W_in, conv, pool=2):
    """
    Calculate the output shape of a convolutional layer
    """
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    
    H_out = ((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
    W_out = ((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
    
    return int(H_out), int(W_out)

def generate_gradcam(model, input_tensor, class_idx, original_image_path):
    """
    Generates a Grad-CAM heatmap for a given model and input.
    Returns the heatmap as a numpy array.
    """
    model.eval()
    
    # Get the last convolutional layer (conv4 in our architecture)
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'conv4' in name:
            last_conv_layer = module
            break
    
    # If conv4 not found, get the last conv layer
    if last_conv_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module

    if last_conv_layer is None:
        return None

    # Register hooks
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)

    backward_handle = last_conv_layer.register_full_backward_hook(backward_hook)
    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass for the specific class
    model.zero_grad()
    output[:, class_idx].backward()
    
    # Remove hooks
    backward_handle.remove()
    forward_handle.remove()
    
    if not gradients or not activations:
        return None
    
    # Process gradients and activations
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    
    # Weight the channels by corresponding gradients
    for i in range(activations[0].shape[1]):
        activations[0][:, i, :, :] *= pooled_gradients[i]
        
    # Generate heatmap
    heatmap = torch.mean(activations[0], dim=1).squeeze().detach().cpu()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    
    # Normalize heatmap
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)
    
    return heatmap.numpy()

# Define Architecture For CNN_TUMOR Model
class CNN_TUMOR(nn.Module):
    
    # Network Initialisation
    def __init__(self, params):
        
        super(CNN_TUMOR, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

class BrainTumorDetector:
    """
    Wrapper class for brain tumor detection model
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'classification_brain_tumor_model', 'Brain_Tumor_model.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.class_labels = {0: 'Brain Tumor', 1: 'Healthy'}
        
    def load_model(self, model_path):
        """Load the trained model."""
        
        # This is the definitive fix. We are temporarily adding the CNN_TUMOR class
        # to the __main__ module's namespace. This is where torch.load looks for the 
        # class definition when the model was saved from a standalone script.
        import sys
        sys.modules['__main__'].CNN_TUMOR = CNN_TUMOR

        print(f"Attempting to load model from: {model_path}")
        
        model = None
        try:
            # Load the model file. PyTorch will now be able to find CNN_TUMOR.
            model = torch.load(model_path, map_location=self.device)
            print(f"Model loaded successfully: {type(model)}")
        
        except Exception as e:
            print(f"FATAL: Failed to load model with monkey-patch: {e}")
            # Clean up the temporary change to __main__ before re-raising the error.
            del sys.modules['__main__'].CNN_TUMOR
            raise e

        # Clean up the temporary change to the __main__ module.
        del sys.modules['__main__'].CNN_TUMOR
        
        # In case the saved file was a state_dict instead of a full model object.
        if isinstance(model, dict):
             print("Loaded file is a state_dict. Creating new model and loading weights.")
             params_model = {
                "shape_in": (3, 256, 256),
                "initial_filters": 16,
                "num_fc1": 100,
                "dropout_rate": 0.25,
                "num_classes": 2
             }
             state_dict = model
             model = CNN_TUMOR(params_model)
             model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model
    
    def get_transform(self):
        """Get the image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Make prediction on an image and generate Grad-CAM heatmap for all predictions."""
        try:
            # Load and preprocess image
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            result = {
                'prediction': self.class_labels[predicted_class],
                'confidence': confidence * 100,
                'class_id': predicted_class,
                'probabilities': {
                    'Brain Tumor': probabilities[0][0].item() * 100,
                    'Healthy': probabilities[0][1].item() * 100
                },
                'heatmap_url': None
            }
            
            # Generate heatmap for the predicted class
            heatmap = generate_gradcam(self.model, image_tensor, class_idx=predicted_class, original_image_path=image_path)
            
            if heatmap is not None:
                # Load original image for overlay
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize heatmap to match image dimensions
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                
                # Apply color map based on prediction
                if predicted_class == 0:  # Brain Tumor - Red to Yellow
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_HOT)
                else:  # Healthy - Blue to Cyan
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_COOL)
                
                # Convert back to RGB for proper overlay
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # Overlay heatmap on original image
                alpha = 0.6  # Transparency factor
                superimposed_img = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
                
                # Convert back to BGR for saving
                superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
                
                # Save the heatmap image
                fs = FileSystemStorage()
                heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
                heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_filename)
                cv2.imwrite(heatmap_path, superimposed_img_bgr)
                result['heatmap_url'] = fs.url(heatmap_filename)
            
            return result

        except Exception as e:
            import traceback
            error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print detailed error for debugging
            
            # Check if the model is properly loaded
            if not hasattr(self, 'model') or self.model is None:
                return {'error': "Model not properly loaded. Please restart the server."}
                
            # Check if it's an image loading error
            if "cannot identify image file" in str(e):
                return {'error': "Cannot process the image. Please try a different image format."}
                
            return {'error': f"Error during prediction: {str(e)}"}


class AnalysisHistory(models.Model):
    """Model to store analysis history"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='analysis_images/')
    heatmap_image = models.ImageField(upload_to='heatmap_images/', null=True, blank=True)
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    tumor_probability = models.FloatField()
    healthy_probability = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.prediction} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_image_filename(self):
        return os.path.basename(self.image.name)
    
    def get_report_data(self):
        """Get data for report generation"""
        return {
            'id': str(self.id),
            'filename': self.get_image_filename(),
            'prediction': self.prediction,
            'confidence': f"{self.confidence:.1f}%",
            'tumor_probability': f"{self.tumor_probability:.1f}%",
            'healthy_probability': f"{self.healthy_probability:.1f}%",
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'image_url': self.image.url
        }
