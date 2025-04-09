import os
from typing import List, Optional
import logging
from concurrent.futures import ProcessPoolExecutor

import fitz  # PyMuPDF
from google.cloud import vision
from pdf2image import convert_from_path
import pandas as pd
import tabula
from PIL import Image
import io

from config import Config

logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

config = Config()

class PDFProcessor:
    def __init__(self, output_format: str = 'txt'):
        self.output_format = output_format

    def set_output_format(self, output_format: str) -> None:
        self.output_format = output_format

    @staticmethod
    def pil_to_bytes(image: Image.Image) -> bytes:
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='PNG')
        byte_stream.seek(0)
        return byte_stream.read()

    @staticmethod
    def get_num_pages(filename: str) -> int:
        with fitz.open(filename) as doc:
            return len(doc)


    def process_image(self, image: Image.Image) -> List[str]:
        client = vision.ImageAnnotatorClient()
        try:
            img_bytes = self.pil_to_bytes(image)
            vision_image = vision.Image(content=img_bytes)
            response = client.document_text_detection(image=vision_image)
            text = response.full_text_annotation.text
            return [line.strip() for line in text.split('\n') if line.strip()]
        except Exception as e:
            logger.error(f'Error during OCR: {str(e)}')
            return []
        
        
    
    def process_pdf(self, filename: str) -> Optional[str]:
        try:
            images = convert_from_path(filename)
        except Exception as e:
            logger.error(f'Error during PDF conversion: {str(e)}')
            raise

        data = []

        with ProcessPoolExecutor() as executor:
            # Map the process_image function directly to images
            results = list(executor.map(self.process_image, images))

        # Flatten the list of results
        data = [item for sublist in results for item in sublist]            

        df = pd.DataFrame(data, columns=['Text'])

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        txt_filename = f'{base_filename}_text.txt'
        txt_filepath = os.path.join('outputs', txt_filename)

        try:
            df.to_csv(txt_filepath, sep='\t', index=False)
            return txt_filename
        except Exception as e:
            logger.error(f'Error during writing to file: {str(e)}')
            return None    

    def extract_tables(self, filename: str) -> List[str]:
        try:
            dfs = tabula.read_pdf(filename, pages='all')
        except Exception as e:
            logger.error(f'Error reading PDF: {e}')
            return []

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        csv_filenames = []

        for i, table in enumerate(dfs):
            csv_filename = f'{base_filename}_table_{i}.csv'
            csv_filepath = os.path.join('outputs', csv_filename)
            try:
                table.to_csv(csv_filepath, index=False)
                csv_filenames.append(csv_filename)
            except Exception as e:
                logger.error(f"Error saving table {i}: {e}")

        return csv_filenames

    def extract_images(self, filename: str, perform_ocr: bool) -> List[str]:
        image_filenames = []

        with fitz.open(filename) as doc:
            base_filename = os.path.splitext(os.path.basename(filename))[0]

            for page_num in range(len(doc)):
                for img_index, img in enumerate(doc.get_page_images(page_num)):
                    try:
                        xref = img[0]
                        img_data = doc.extract_image(xref)
                        pil_image = Image.open(io.BytesIO(img_data['image']))

                        image_filename = f'{base_filename}_{page_num}_img{img_index}.png'
                        image_filepath = os.path.join('outputs', image_filename)
                        pil_image.save(image_filepath)
                        image_filenames.append(image_filename)

                        if perform_ocr:
                            self.perform_ocr_on_image(pil_image, base_filename, page_num, img_index)

                    except Exception as e:
                        logger.error(f'Error processing image on page {page_num}: {str(e)}')

        return image_filenames

    def perform_ocr_on_image(self, pil_image: Image.Image, base_filename: str, page_num: int, img_index: int) -> None:
        client = vision.ImageAnnotatorClient()
        try:
            img_data_bytes = self.pil_to_bytes(pil_image)
            vision_image = vision.Image(content=img_data_bytes)
            response = client.document_text_detection(image=vision_image)
            text = response.full_text_annotation.text

            ocr_txt_filename = f'{base_filename}_{page_num}_img{img_index}_ocr.txt'
            ocr_txt_filepath = os.path.join('outputs', ocr_txt_filename)
            with open(ocr_txt_filepath, 'w') as f:
                f.write(text)
        except Exception as e:
            logger.error(f'Error during OCR for image {img_index} on page {page_num}: {str(e)}')
        

    def process_pdf_file(self, filename: str) -> Optional[str]:
        txt_filename = self.process_pdf(filename)
        if txt_filename:
            print(f"Done, text saved as {txt_filename}")
        else:
            print("An error occurred. Check console for details.")
        return txt_filename

    def process_pdf_for_tables(self, filename: str) -> List[str]:
        csv_filenames = self.extract_tables(filename)
        if csv_filenames:
            print(f"Done, tables saved as {', '.join(csv_filenames)}")
        else:
            print("No tables found.")
        return csv_filenames

    def process_pdf_for_images(self, filename: str, perform_ocr: bool) -> List[str]:
        image_filenames = self.extract_images(filename, perform_ocr)
        if image_filenames:
            print(f"Done, images saved in directory {'outputs/'+ os.path.splitext(os.path.basename(filename))[0]}")
        else:
            print("No images found.")
        return image_filenames