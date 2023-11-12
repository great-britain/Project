import easyocr
import os
from shutil import move
from PIL import Image

def classify_images(input_folder, output_folders, reader):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)

            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f'Невозможно обработать изображение {filename}: {e}')
                continue

            result = reader.readtext(file_path)
            text_detected = " ".join([detection[1] for detection in result])

            for key in output_folders:
                if key in text_detected:
                    move(file_path, os.path.join(output_folders[key], filename))
                    print(f'Изображение {filename} перемещено в папку {output_folders[key]}')
                    break  # Прерывание цикла после первого совпадения

input_folder = 'input_folder'
output_folders = {
    'DpR': 'output_folder_DpR',
    'com2': 'output_folder_Pgpcom2',
    'comz': 'output_folder_Pgpcom2',
    'pc': 'output_folder_Pgplpc2',
    'com3': 'output_folder_Phlcom3',
    'K3-8': 'output_folder_PhpK3-8',
    'K3-1': 'output_folder_PhpK3-1',
    'Ctm-K': 'output_folder_Php-Ctm-K-1',
    'Ctm-Shv1': 'output_folder_Php-Ctm-Shv1',
    'nta': 'output_folder_Php-nta4',
    'K1-3-3-5': 'output_folder_Spp-210-3-3-5',
    'K1-3-3-6': 'output_folder_Spp-210-3-3-6',
    'Spp-K1': 'output_folder_Spp-K1'

}

reader = easyocr.Reader(['en'])  # Инициализация только один раз

# Создание выходных папок
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

classify_images(input_folder, output_folders, reader)
