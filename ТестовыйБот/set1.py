from ultralytics import YOLO
import cv2

import os
import glob
import re

import numpy as np


class ModelDetected:
    """Класс детекции нахождения персонала в опасных зонах"""

    def __init__(self,
                 model: YOLO,
                 conf_level: float = 0.3,
                 verbose: bool = False,
                 device: str = 'cpu'
                 ):
        self.scale_percent = 50
        self.conf_level = conf_level
        self.iou = 0.5
        self.verbose = verbose
        self.device = device
        self.input_dir = None
        self.classes = [0]  # 0: 'person'
        # Загружаем модель
        self.model = model
        # Словарь опасных зон, с указанием координат
        self.danger_zones = {}
        # Словарь фотографий, с указанием зон
        self.photos_by_zone = {}

    def load_danger_zones(self, path_zones: str):
        """Загружаем координаты опасных зон
            Заполняется словарь self.danger_zones - Словарь опасных зон, с указанием координат
        :param path_zones - путь к списку координат зон в формате *.txt
        """
        files_danger_zones = glob.glob(path_zones + "/*.txt")
        for fname in files_danger_zones:
            zone_name = fname.strip().split('\\')[-1].split('/')[-1].split('.')[0]
            with open(fname, 'r') as f:
                coords = f.read()
            coords = [list(map(int, re.findall(r'\d+', coord))) for coord in re.findall(r'\[.+?\]', coords)]
            # self.danger_zones[zone_name] = coords
            self.danger_zones[zone_name] = np.array(coords, np.int32)

    def load_photos(self, path_cameras: str, file_types: tuple = ('*.jpg', '*.jpeg', '*.png', '*.gif')):
        """Загружаем фотографии для детекции
            Заполняется словарь self.photos_by_zone - Словарь фотографий, с указанием зон
        :param path_cameras - путь к файлам. Директория должна содержать поддиректории опасных зон
        """

        pathes_zones = [f.path for f in os.scandir(path_cameras) if f.is_dir()]
        for dir_zone in pathes_zones:
            zone_name = dir_zone.strip().split('\\')[-1].split('/')[-1].split('.')[0]
            if zone_name not in self.photos_by_zone:
                self.photos_by_zone[zone_name] = []
            for filetype in file_types:
                files_photos = glob.glob(dir_zone + f"/{filetype}")
                self.photos_by_zone[zone_name].extend(files_photos)

    def model_predict(self, img_files: list, zone_name: str, output_dir_zone: str):
        self.alpha = 0.4
        pts = self.danger_zones[f"danger_{zone_name}"].reshape((-1, 1, 2))

        for imgage_file in img_files:
            file_name = imgage_file.strip().split('\\')[-1].split('/')[-1]
            input_image = cv2.imread(imgage_file)

            result = self.model.predict(input_image, conf=self.conf_level, iou=self.iou, classes=self.classes,
                                        device=self.device, verbose=False)

            # Получаем боундбоксы и сегменты
            boxes = result[0].boxes.xyxy.cpu().numpy()
            confs = result[0].boxes.conf.cpu().numpy()
            classes = result[0].boxes.cls.cpu().numpy()

            is_segments = result[0].masks is not None
            if is_segments:
                segments = result[0].masks.xy
            # Рисуем опасную зону
            danger_zone_image = input_image.copy()
            cv2.polylines(danger_zone_image, pts=[pts], isClosed=True, color=(224, 128, 67), thickness=2)
            cv2.fillPoly(danger_zone_image, [pts], (213, 107, 57))
            input_image = cv2.addWeighted(danger_zone_image, self.alpha, input_image, 1 - self.alpha, 0)
            cv2.polylines(input_image, pts=[pts], isClosed=True, color=(224, 128, 67), thickness=2)

            for idx in range(len(boxes)):
                box = boxes[idx].astype('int32')
                confidence = confs[idx]
                if is_segments:
                    segment = segments[idx].astype('int32')
                detect_class = classes[idx]
                xmin, ymin, xmax, ymax = box.astype('int')

                intersection = 0.83
                helmet = 0.53

                category = classes[idx].astype('int')
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                # Рисуем boundbox
                cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (52, 121, 39), 2)  # box
                cv2.putText(img=input_image, text=f'Person : {int(confidence * 100)}%',
                            org=(xmin, ymin - 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(15, 41, 7),
                            thickness=1)
                cv2.putText(img=input_image, text=f'Warning: {int(intersection * 100)}%',
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(15, 41, 7),
                            thickness=1)

                if is_segments:
                    # Рисуем сегментацию
                    segment_image = input_image.copy()
                    cv2.polylines(img=segment_image, pts=[segment], isClosed=True, color=(30, 129, 176), thickness=2)
                    cv2.fillPoly(segment_image, pts=[segment], color=(37, 150, 190))
                    input_image = cv2.addWeighted(segment_image, self.alpha, input_image, 1 - self.alpha, 0)
                    cv2.polylines(img=input_image, pts=[segment], isClosed=True, color=(30, 129, 176), thickness=2)

                # Выводим итоговое изображение
                # plt.figure(figsize = (20,22))
                # plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
                # Сохрнаяем итоговое изображение
                cv2.imwrite(output_dir_zone + "/result_" + file_name, input_image)
                # Сохрнаяем файл со статистикаой по изображению
                info_predict = [file_name, confidence, intersection, helmet]
                info_predict = list(map(str, info_predict))

                with open(output_dir_zone + "/result_" + file_name.split(".")[0] + ".txt", 'w') as txt_file:
                    txt_file.write("\t".join(info_predict))

    def detected(self, output_dir: str):
        for zone in self.photos_by_zone.keys():
            output_dir_zone = f"{output_dir}/{zone}"
            # Создаем выходную директорию для зоны если её нет
            os.makedirs(output_dir_zone, exist_ok=True)
            for file_photo in self.photos_by_zone[zone]:
                # print(file_photo)
                self.model_predict(img_files=[file_photo], zone_name=zone, output_dir_zone=output_dir_zone)

    def process_single_image(self, image_path, output_path, camera_name):
        """
        Функция для обработки одного изображения.
        :param image_path: Путь к изображению для обработки.
        :param output_path: Путь для сохранения обработанного изображения.
        :param camera_name: Имя камеры, которое используется для определения зоны.
        :return: Путь к обработанному изображению.
        """
        # Путь
        model_seg = YOLO("yolov8x-seg.pt")


        # Замена'some_zone' на переданное имя камеры
        processed_image_path = self.model_predict(img_files=[image_path], zone_name=camera_name,
                                                  output_dir_zone=output_path)

        # Предположим, что обработанное изображение сохраняется в output_path
        return processed_image_path


if __name__ == '__main__':
    # Пути
    PATH = ''
    # Путь к данным
    DATASET_PATH = PATH + 'mini_train_dataset_train/'
    # Путь к опасным зонам
    DANGER_ZONES_PATH = DATASET_PATH + 'danger_zones/'
    # Путь к фотографиям
    CAMERAS_PATH = DATASET_PATH + 'cameras/'
    # Путь выходных данных
    OUTPUT_PATH = PATH + 'output/'

    # Создаем класс детекции
    # device = 'cuda'
    device = 'cpu'
    # Загружаем модель
    # model = YOLO('yolov8x.pt')
    model_seg = YOLO("yolov8x-seg.pt")

    detector = ModelDetected(model=model_seg, device=device)
    # Опредлеям доступные типы изображений
    file_types = ('*.jpg', '*.jpeg', '*.png', '*.gif')
    # Загружаем опасные зоны
    detector.load_danger_zones(path_zones=DANGER_ZONES_PATH)
    # Загружаем фотографии для анализа
    detector.load_photos(path_cameras=CAMERAS_PATH, file_types=file_types)
    # Производим детекцию, результат сохраняется в OUTPUT_PATH
    detector.detected(output_dir=OUTPUT_PATH)
