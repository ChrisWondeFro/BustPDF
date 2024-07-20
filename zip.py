
import os
import shutil

def compress_directory(directory, output_path=None):
    try:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            return

        if not output_path:
            output_path = os.getcwd()

        output_file = os.path.join(output_path, os.path.basename(directory))

        if os.path.exists(output_file + '.zip'):
            print(f"Error: {output_file}.zip already exists.")
            return

        shutil.make_archive(output_file, 'zip', directory)
        print(f'Successfully compressed {directory} into {output_file}.zip')
    except Exception as e:
        print(f"An error occurred: {e}")

directory = '/Users/christian/Desktop/BustPDF'
output_directory = '/Users/christian/Desktop'
compress_directory(directory, output_directory)
