from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.template.loader import render_to_string
from django.conf import settings
import os
import json
from .models import BrainTumorDetector, AnalysisHistory
import logging
from datetime import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Set up logging
logger = logging.getLogger(__name__)

# Initialize the model (load once when the app starts)
try:
    detector = BrainTumorDetector()
    logger.info("Brain tumor detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    detector = None

# --- Segmentation Model Loader ---
from .models import SegmentationBrainTumorModel
SEG_MODEL_PATH = os.path.join(settings.BASE_DIR, 'segmentation_braintumor_model', 'model.h5')
try:
    segmentation_model = SegmentationBrainTumorModel(SEG_MODEL_PATH)
    logger.info("Segmentation model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load segmentation model: {e}")
    segmentation_model = None

def home(request):
    """Home page view"""
    return render(request, 'tumor_detector/home.html')

@csrf_exempt
def predict(request):
    """Handle image upload and prediction"""
    if request.method == 'POST':
        try:
            # Check if we're reanalyzing an existing image
            if 'analysis_id' in request.POST:
                try:
                    analysis = AnalysisHistory.objects.get(id=request.POST['analysis_id'])
                    file_path = analysis.image.path
                    
                    # Make prediction on the saved image
                    if detector is None:
                        return JsonResponse({'error': 'Model not loaded. Please try again later.'}, status=500)
                    
                    result = detector.predict(file_path)
                    
                    if 'error' in result:
                        return JsonResponse({'error': result['error']}, status=500)
                    
                    # Update the analysis record
                    analysis.prediction = result['prediction']
                    analysis.confidence = result['confidence']
                    analysis.tumor_probability = result['probabilities']['Brain Tumor']
                    analysis.healthy_probability = result['probabilities']['Healthy']
                    
                    # Update heatmap if available
                    if result.get('heatmap_url'):
                        analysis.heatmap_image = result['heatmap_url']
                    
                    analysis.save()
                    
                    # Add analysis ID and image URL to the result
                    result['analysis_id'] = str(analysis.id)
                    result['image_url'] = analysis.image.url
                    
                    return JsonResponse(result)
                except AnalysisHistory.DoesNotExist:
                    return JsonResponse({'error': 'Analysis not found'}, status=404)
            
            # Check if file was uploaded for new analysis
            if 'image' not in request.FILES:
                return JsonResponse({'error': 'No image file provided'}, status=400)
            
            image_file = request.FILES['image']
            
            # Validate file type
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
            if image_file.content_type not in allowed_types:
                return JsonResponse({'error': 'Invalid file type. Please upload a JPEG, PNG, or BMP image.'}, status=400)
            
            # Save the uploaded file temporarily
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_file_url = fs.url(filename)
            file_path = fs.path(filename)
            
            # Make prediction
            if detector is None:
                return JsonResponse({'error': 'Model not loaded. Please try again later.'}, status=500)
            
            result = detector.predict(file_path)
            
            if 'error' in result:
                # Clean up the uploaded file on error
                try:
                    os.remove(file_path)
                except:
                    pass
                return JsonResponse({'error': result['error']}, status=500)
            
            # Save analysis to history
            try:
                analysis = AnalysisHistory.objects.create(
                    image=image_file,
                    prediction=result['prediction'],
                    confidence=result['confidence'],
                    tumor_probability=result['probabilities']['Brain Tumor'],
                    healthy_probability=result['probabilities']['Healthy'],
                    heatmap_image=result.get('heatmap_url') # Use .get() for safety
                )
                result['analysis_id'] = str(analysis.id)
            except Exception as e:
                logger.error(f"Failed to save analysis history: {e}")
                # Continue without saving history if it fails
            
            # Clean up the temporary file
            try:
                os.remove(file_path)
            except:
                pass
            
            # Add the uploaded image URL to the result
            result['image_url'] = uploaded_file_url
            
            return JsonResponse(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return JsonResponse({'error': 'An error occurred during prediction'}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


@csrf_exempt
def segment(request):
    """Handle image upload and segmentation prediction"""
    if request.method == 'POST':
        try:
            color_map = request.POST.get('color_map', 'green')
            logger.info(f"SEGMENT: color_map={color_map}, POST={dict(request.POST)}, FILES={list(request.FILES.keys())}")
            if 'image' not in request.FILES and 'analysis_id' not in request.POST:
                logger.error("SEGMENT: No image file or analysis_id provided")
                return JsonResponse({'error': 'No image file provided'}, status=400)
            # Support reanalysis by analysis_id
            if 'analysis_id' in request.POST:
                from .models import AnalysisHistory
                try:
                    analysis = AnalysisHistory.objects.get(id=request.POST['analysis_id'])
                    file_path = analysis.image.path
                    logger.info(f"SEGMENT: Using file_path from analysis: {file_path}")
                except AnalysisHistory.DoesNotExist:
                    logger.error("SEGMENT: Analysis not found for id %s", request.POST['analysis_id'])
                    return JsonResponse({'error': 'Analysis not found'}, status=404)
            else:
                image_file = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image_file.name, image_file)
                file_path = fs.path(filename)
                logger.info(f"SEGMENT: Uploaded new image, file_path={file_path}")
            # Check file existence
            if not os.path.exists(file_path):
                logger.error(f"SEGMENT: File does not exist: {file_path}")
                return JsonResponse({'error': f'File does not exist: {file_path}'}, status=400)
            if segmentation_model is None:
                logger.error(f"SEGMENT: Segmentation model not loaded. Model path: {SEG_MODEL_PATH}")
                return JsonResponse({'error': 'Segmentation model not loaded. Please try again later.'}, status=500)
            try:
                # Predict mask
                mask = segmentation_model.predict_mask(file_path)
                tumor_area = int(mask.sum())
                MIN_TUMOR_AREA = 100  # You can adjust this threshold
                if tumor_area < MIN_TUMOR_AREA:
                    if 'analysis_id' not in request.POST:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"SEGMENT: Could not remove file: {file_path}, error: {e}")
                    return JsonResponse({
                        'overlay_url': None,
                        'tumor_area': tumor_area,
                        'prediction': 'Healthy (no tumor detected)'
                    })
                # Overlay mask with color map
                overlay = segmentation_model.overlay_mask(file_path, mask, color_map=color_map)
                overlay_filename = f"seg_overlay_{os.path.basename(file_path)}"
                overlay_path = os.path.join(settings.MEDIA_ROOT, overlay_filename)
                import cv2
                cv2.imwrite(overlay_path, overlay)
                fs = FileSystemStorage()
                overlay_url = fs.url(overlay_filename)
                if 'analysis_id' not in request.POST:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"SEGMENT: Could not remove file: {file_path}, error: {e}")
                return JsonResponse({
                    'overlay_url': overlay_url,
                    'tumor_area': tumor_area,
                    'prediction': 'Tumor detected' if tumor_area >= MIN_TUMOR_AREA else 'Healthy (no tumor detected)'
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"SEGMENT: Exception during segmentation: {e}\n{tb}")
                return JsonResponse({'error': f'Exception during segmentation: {str(e)}', 'traceback': tb}, status=500)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"SEGMENT: Outer exception: {e}\n{tb}")
            return JsonResponse({'error': f'Outer exception: {str(e)}', 'traceback': tb}, status=500)
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def get_history(request):
    """Get analysis history"""
    try:
        # Get recent analyses (last 20)
        analyses = AnalysisHistory.objects.all()[:20]
        history_data = []
        
        for analysis in analyses:
            history_data.append({
                'id': str(analysis.id),
                'image_url': analysis.image.url,
                'heatmap_url': analysis.heatmap_image.url if analysis.heatmap_image else None,
                'prediction': analysis.prediction,
                'confidence': analysis.confidence,
                'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M'),
                'filename': analysis.get_image_filename()
            })
        
        return JsonResponse({'history': history_data})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return JsonResponse({'error': 'Failed to get history'}, status=500)


def reanalyze(request, analysis_id):
    """Reanalyze a previous image"""
    try:
        analysis = AnalysisHistory.objects.get(id=analysis_id)
        
        # Make prediction on the saved image
        if detector is None:
            return JsonResponse({'error': 'Model not loaded. Please try again later.'}, status=500)
        
        result = detector.predict(analysis.image.path)
        
        if 'error' in result:
            return JsonResponse({'error': result['error']}, status=500)
        
        # Update the analysis record
        analysis.prediction = result['prediction']
        analysis.confidence = result['confidence']
        analysis.tumor_probability = result['probabilities']['Brain Tumor']
        analysis.healthy_probability = result['probabilities']['Healthy']
        
        # Update heatmap if available
        if result.get('heatmap_url'):
            analysis.heatmap_image = result['heatmap_url']
        
        analysis.save()
        
        result['image_url'] = analysis.image.url
        result['analysis_id'] = str(analysis.id)
        
        return JsonResponse(result)
        
    except AnalysisHistory.DoesNotExist:
        return JsonResponse({'error': 'Analysis not found'}, status=404)
    except Exception as e:
        logger.error(f"Reanalysis error: {e}")
        return JsonResponse({'error': 'An error occurred during reanalysis'}, status=500)


def _ensure_absolute_media_path(path):
    import os
    from django.conf import settings
    if not path:
        return None
    if path.startswith(settings.MEDIA_URL):
        abs_path = os.path.join(settings.MEDIA_ROOT, path[len(settings.MEDIA_URL):].lstrip('/'))
        return abs_path
    return path

def _get_absolute_image_path(image_field_or_url):
    import os
    from django.conf import settings
    # If it's a Django File/ImageField
    if hasattr(image_field_or_url, 'path'):
        return image_field_or_url.path
    # If it's a string
    if isinstance(image_field_or_url, str):
        # If it's already an absolute path and exists
        if os.path.isabs(image_field_or_url) and os.path.exists(image_field_or_url):
            return image_field_or_url
        # If it's a media URL
        if image_field_or_url.startswith(settings.MEDIA_URL):
            abs_path = os.path.join(settings.MEDIA_ROOT, image_field_or_url[len(settings.MEDIA_URL):].lstrip('/'))
            if os.path.exists(abs_path):
                return abs_path
    return None

def download_report(request, analysis_id):
    """Generate and download a PDF report"""
    try:
        analysis = AnalysisHistory.objects.get(id=analysis_id)
        
        # Create PDF
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="brain_tumor_analysis_{analysis_id}.pdf"'
        
        # Create the PDF object
        doc = SimpleDocTemplate(response, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Brain Tumor Detection Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Analysis details
        story.append(Paragraph(f"<b>Analysis ID:</b> {analysis_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Image:</b> {analysis.get_image_filename()}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Results table
        data = [
            ['Parameter', 'Value'],
            ['Prediction', analysis.prediction],
            ['Confidence', f"{analysis.confidence:.1f}%"],
            ['Brain Tumor Probability', f"{analysis.tumor_probability:.1f}%"],
            ['Healthy Probability', f"{analysis.healthy_probability:.1f}%"]
        ]
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # --- Add Images Section ---
        from reportlab.lib.utils import ImageReader
        import os
        story.append(Paragraph("<b>Analyzed Image(s):</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        # 1. Original image
        orig_path = _get_absolute_image_path(analysis.image)
        logger.info(f"REPORT: Original image path used: {orig_path}")
        if orig_path and os.path.exists(orig_path):
            story.append(Paragraph("Original Uploaded Image:", styles['Normal']))
            story.append(Spacer(1, 4))
            story.append(RLImage(orig_path, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
        # 2. Heatmap (classification)
        heatmap_path = _get_absolute_image_path(analysis.heatmap_image)
        logger.info(f"REPORT: Heatmap image path used: {heatmap_path}")
        if heatmap_path and os.path.exists(heatmap_path):
            story.append(Paragraph("Classification Heatmap:", styles['Normal']))
            story.append(Spacer(1, 4))
            story.append(RLImage(heatmap_path, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
        # 3. Segmentation overlay (if exists)
        base_image_name = os.path.basename(orig_path) if orig_path else None
        seg_overlay_candidates = [
            f"seg_overlay_{base_image_name}" if base_image_name else None,
            f"seg_overlay_{os.path.splitext(base_image_name)[0]}.jpg" if base_image_name else None,
            f"seg_overlay_{os.path.splitext(base_image_name)[0]}.png" if base_image_name else None
        ]
        media_dir = os.path.join(settings.MEDIA_ROOT)
        found_overlay = None
        for candidate in seg_overlay_candidates:
            if not candidate:
                continue
            candidate_path = os.path.join(media_dir, candidate)
            if os.path.exists(candidate_path):
                found_overlay = candidate_path
                break
        logger.info(f"REPORT: Segmentation overlay path used: {found_overlay}")
        if found_overlay:
            story.append(Paragraph("Segmentation Overlay:", styles['Normal']))
            story.append(Spacer(1, 4))
            story.append(RLImage(found_overlay, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey
        )
        story.append(Paragraph("Disclaimer: This analysis is for educational and research purposes only. Always consult with qualified medical professionals for diagnosis and treatment.", disclaimer_style))
        
        # Build PDF
        doc.build(story)
        return response
        
    except AnalysisHistory.DoesNotExist:
        return JsonResponse({'error': 'Analysis not found'}, status=404)
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return JsonResponse({'error': 'Failed to generate report'}, status=500)

@csrf_exempt
def delete_analysis(request, analysis_id):
    """Delete an analysis from history"""
    try:
        analysis = AnalysisHistory.objects.get(id=analysis_id)
        
        # Delete the image files
        try:
            if analysis.image:
                if os.path.exists(analysis.image.path):
                    os.remove(analysis.image.path)
            
            if analysis.heatmap_image:
                if os.path.exists(analysis.heatmap_image.path):
                    os.remove(analysis.heatmap_image.path)
        except Exception as e:
            logger.error(f"Error deleting image files: {e}")
            # Continue with deletion even if file removal fails
        
        # Delete the database record
        analysis.delete()
        
        return JsonResponse({'success': True, 'message': 'Analysis deleted successfully'})
        
    except AnalysisHistory.DoesNotExist:
        return JsonResponse({'error': 'Analysis not found'}, status=404)
    except Exception as e:
        logger.error(f"Delete analysis error: {e}")
        return JsonResponse({'error': 'An error occurred while deleting the analysis'}, status=500)
