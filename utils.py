
import google.cloud.vision as vision

from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import fitz # PyMuPDF
import io
from PIL import Image  
import torch
from torchvision import models, transforms  
import tabula as tabula_py
import os
import firebase_admin
from firebase_admin import credentials
from config import Config
import logging


logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger=logging.getLogger(__name__)

config = Config()


class PDFProcessor:
    def __init__(self, model_name=('resnet50'), output_format='txt'):
        self.output_format = output_format
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #if model_name == 'resnet50':
            #self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        #else:
            #self.model = None  # Handle other models as required
        #if self.model:
            #self.model.eval()

        # Initialize the Google Vision API client
        self.client = vision.ImageAnnotatorClient()

        # Define your class names here if needed
        self.class_names = ['graph', 'photo', 'diagram', 'table', 'text', 'text_messages'] 

    def zipdir(self, path, ziph):
    # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                          os.path.join(path, '..')))


    def set_output_format(self, output_format):
        self.output_format = output_format  

    def pil_to_bytes(self, image: Image) -> bytes:
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_stream.seek(0)
        return byte_stream.read()
    
    def get_num_pages(self, filename):
        # Return the number of pages in the PDF
        doc = fitz.open(filename)
        num_pages = len(doc)
        doc.close()
        return num_pages
    
    def process_pdf(self, filename):
        # Convert pdf into list of images in chunks
        try:
            images = convert_from_path(filename)         
        except Exception as e:
            logger.error(f'Error during PDF conversion: {str(e)}')
            raise e  
        data = []

        # Perform OCR on the images
        for i in range(len(images)):
            try:
                text = pytesseract.image_to_string(images[i], lang='eng')
                # Process the text as needed
                lines = text.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = ' '.join(line.split())
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)
                data.extend(cleaned_lines)
            except Exception as e:
                print(f'Error during OCR: {str(e)}')
                return None    

        # Convert the data to a DataFrame
        df = pd.DataFrame(data, columns=['Text'])

        # Write DataFrame to txt
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        txt_filename = base_filename + '_text.txt'
        txt_filepath = os.path.join('outputs', txt_filename)
        try:
            df.to_csv(txt_filepath, sep='\t', index=False)
        except Exception as e:
            print(f'Error during writing to file: {str(e)}')
            return None  
        return txt_filename
    
    def extract_tables(self, filename):
        try:
            dfs = tabula_py.read_pdf(filename, pages='all')
        except Exception as e:
            logger.error(f'Error reading PDF: {e}')
            return []    

        print(f"Number of tables found: {len(dfs)}")

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        csv_filenames = []
        for i, table in enumerate(dfs):
            csv_filename = base_filename + f'_table_{i}.csv'
            csv_filepath = os.path.join('outputs', csv_filename)
            try:
                table.to_csv(csv_filepath, index=False)
                csv_filenames.append(csv_filename)
                print(f"Table {i} saved to {csv_filename}")
            except Exception as e:
                logger.error(f"Error saving table {i}: {e}")   

        return csv_filenames
    
    def extract_images(self, filename, perform_ocr):
        response = None
        doc = fitz.open(filename)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        image_filenames = []

        try:
            for i in range(len(doc)):
                image_list = doc.get_page_images(i)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    try:
                       img_data = doc.extract_image(xref)
                       pil_image = Image.open(io.BytesIO(img_data['image']))
                    except Exception as e:
                        logger.error(f'Error during image extraction: {str(e)}')
                        continue
                    print(f"Image data type: {type(img_data)}")

                    # Updated naming to align with other methods
                    image_filename = f'{base_filename}_{i}_img{img_index}.png'
                    image_filepath = os.path.join('outputs', image_filename)
                    pil_image.save(image_filepath)
                    image_filenames.append(image_filename)
                    print(f"Attempting to save image to: {image_filepath}")

                    if perform_ocr:
                        try:
                          # Convert the PIL image to bytes and then to a vision.Image
                            img_data_bytes = self.pil_to_bytes(pil_image)
                            vision_image = vision.Image(content=img_data_bytes)
                            response = self.client.annotate_image({'image': vision_image})
                            text = response.text_annotations[0].description if response.text_annotations else ""
                            ocr_txt_filename = f'{base_filename}_{i}_img{img_index}_ocr.txt'
                            ocr_txt_filepath = os.path.join('outputs', ocr_txt_filename)
                            with open(ocr_txt_filepath, 'w') as f:
                                f.write(text)

                        except Exception as e:
                           logger.error(f'Error during OCR in extract_images: {str(e)}')
                           print(f'Error during OCR: {str(e)}')
                           continue

                    # If you have an image classification model and class names defined
                if self.model and self.class_names:
                    try:
                        pil_image_rgb = pil_image.convert('RGB')
                        tensor_image = self.transform(pil_image_rgb)
                        tensor_image_rgb = tensor_image.unsqueeze(0)
                        with torch.no_grad():
                            output = self.model(tensor_image_rgb)
                            _, predicted = output.max(1)
                            predicted_index = predicted.item()
                            # Check if the predicted index is within the valid range
                            if 0 <= predicted_index < len(self.class_names):
                                predicted_class = self.class_names[predicted_index]
                                print(f'Predicted class for image {image_filename}: {predicted_class}')
                            else:
                                logger.error(f'Predicted index {predicted_index} is out of range for class_names.')
                    except Exception as e:
                        logger.error(f'Error during image classification: {str(e)}')

                
        except Exception as e:
            logger.error(f'Error during image extraction: {str(e)}')
            print(f'Error during image extraction: {str(e)}')
        
        return image_filenames
    
    def process_pdf_file(self, filename):
        txt_filename = self.process_pdf(filename)
        if txt_filename:
            print(f"Done, text saved as {txt_filename}")
        else:
            print("An error occurred. Check console for details.")
        return txt_filename

    def process_pdf_for_tables(self, filename):
        csv_filenames = self.extract_tables(filename)
        if csv_filenames:
            print(f"Done, tables saved as {', '.join(csv_filenames)}")
        else:
            print("No tables found.")
        return csv_filenames

    def process_pdf_for_images(self, filename, perform_ocr):
        image_filenames = self.extract_images(filename, perform_ocr)
        if image_filenames:
            print(f"Done, images saved in directory {'outputs/'+ os.path.splitext(os.path.basename(filename))[0]}")
        else :
             print("No images found.") 
             
        return image_filenames

