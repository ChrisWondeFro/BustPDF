
from flask import Flask, Blueprint, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from utils import PDFProcessor
 
import os
import asyncio
import mimetypes

app = Flask(__name__)

pdf_processor = PDFProcessor()

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    ALLOWED_MIME_TYPES = {'application/pdf'}

    # Check the file extension
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Check the MIME type
    try:
        file_mimetype = mimetypes.guess_type(filename)[0]
        if file_mimetype not in ALLOWED_MIME_TYPES:
            return False
    except Exception as e:
        return False

    return True

pdf_bp = Blueprint('pdf', __name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        output_format = request.form['output_format']
        extract_text = 'extract_text' in request.form
        extract_tables = 'extract_tables' in request.form
        extract_images = 'extract_images' in request.form
        perform_ocr = 'perform_ocr' in request.form

        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        pdf_processor.set_output_format(output_format)

        output_files = []
        if extract_text:
            text_filename = pdf_processor.process_pdf(filepath)
            output_files.append(text_filename)

        if extract_tables:
            table_filenames = pdf_processor.extract_tables(filepath)
            output_files.extend(table_filenames)

        if extract_images:
            image_filenames = pdf_processor.extract_images(filepath, perform_ocr)
            output_files.extend(image_filenames)
            print(f"Image filenames from PDFProcessor: {image_filenames}")

        print(f"Output files to be rendered: {output_files}")

        return render_template('results.html', output_files=output_files)

    return render_template('upload.html')

@app.route('/results/')
def results():
    # Get the list of all files in the outputs directory
    output_files = os.listdir('outputs')
    # Prepend the directory name to the filenames
    output_files = ['outputs/' + filename for filename in output_files]

    return render_template('results.html', output_files=output_files)

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download_file(filename):
    return send_from_directory('outputs', filename, as_attachment=True)

app.register_blueprint(pdf_bp)

if __name__ == '__main__':
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    config = Config()
    config.bind = ["localhost:8000"]
    asyncio.run(serve(app, config))

