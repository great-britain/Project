from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import easyocr
import os
import glob
import pandas as pd
import re
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import box, Polygon

TOKEN = '6963573748:AAHzlb7OGiTV3DquOBh8tV3gvpilU8F7svk'

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
        self.data_columns = ['file_name', 'count_persons', 'person', 'warning', 'seg_warning', 'helmet']
        self.result_df = pd.DataFrame(columns=self.data_columns)

    def load_danger_zones(self, path_zones: str):
        """Загружаем координаты опасных зон
            Заполняется словарь self.danger_zones - Словарь опасных зон, с указанием координат
        :param path_zones - путь к списку координат зон в формате *.txt
        """
        files_danger_zones = glob.glob(path_zones + "/*.txt")
        for fname in files_danger_zones:
            zone_name = fname.strip().split('\\')[-1].split('/')[-1].split('.')[0]
            # Для одной камеры может быть несколько опасных зон
            zone_name = re.sub('_zone\d+', '', zone_name)
            with open(fname, 'r') as f:
                coords = f.read()
            coords = [list(map(int, re.findall(r'\d+', coord))) for coord in re.findall(r'\[.+?\]', coords)]
            if zone_name not in self.danger_zones:
                self.danger_zones[zone_name] = []
            self.danger_zones[zone_name].append(np.array(coords, np.int32))

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

    # Расчет пересечения (Intersection), объединения (Union) и IoU
    def intersectionOverUnion(self, pol1_xy, pol2_xy):
        # Опредяем полигоны из набора точек
        polygon1_shape = Polygon(pol1_xy)
        polygon2_shape = Polygon(pol2_xy)

        # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
        # Необходимо для расчета процента нахождения человека в опасной зоне
        polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
        polygon_union = polygon1_shape.union(polygon2_shape).area
        IoU = polygon_intersection / polygon_union
        return polygon_intersection, polygon_union, IoU

    def model_predict(self, img_files: list, zone_name: str, output_dir_zone: str):
        self.alpha = 0.2
        danger_zones = self.danger_zones[f"danger_{zone_name}"]

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
            for danger_zone in danger_zones:
                danger_zones_pts = danger_zone.reshape((-1, 1, 2))
                # Рисуем опасную зону
                danger_zone_image = input_image.copy()
                cv2.polylines(danger_zone_image, pts=[danger_zones_pts], isClosed=True, color=(0, 168, 255),
                              thickness=2)
                cv2.fillPoly(danger_zone_image, pts=[danger_zones_pts], color=(0, 168, 255))
                input_image = cv2.addWeighted(danger_zone_image, self.alpha, input_image, 1 - self.alpha, 0)
                cv2.polylines(input_image, pts=[danger_zones_pts], isClosed=True, color=(0, 168, 255), thickness=2)

            image_stat = {'file_name': file_name, 'count_persons': 0, 'person': 0, 'warning': 0, 'seg_warning': 0,
                          'helmet': 0}

            for idx in range(len(boxes)):
                box = boxes[idx].astype('int32')
                confidence = confs[idx]
                if is_segments:
                    segment = segments[idx].astype('int32')
                detect_class = classes[idx]
                xmin, ymin, xmax, ymax = box.astype('int')

                person_polygon = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
                person_warning = 0
                segment_person_warning = 0
                for danger_zone in danger_zones:
                    # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
                    polygon_intersection, polygon_union, IoU = self.intersectionOverUnion(person_polygon, danger_zone)
                    # Расчитываем процент нахождения человека в опасной зоне
                    person_warning = max(person_warning, polygon_intersection / Polygon(person_polygon).area)
                    if is_segments:
                        # Расчитываем пересечение (Intersection) и объединение (Union) и IOU,
                        segment_polygon_intersection, _, _ = self.intersectionOverUnion(segment, danger_zone)
                        # Расчитываем процент нахождения человека в опасной зоне по сегменту
                        segment_person_warning = max(segment_person_warning,
                                                     segment_polygon_intersection / Polygon(segment).area)

                #                     print(f"polygon_intersection: {polygon_intersection}")
                #                     print(f"polygon_union: {polygon_union}")
                #                     print(f"IoU: {IoU}")
                #                     print(f"person warning: {person_warning}")
                #                     print(f"person seg_warning: {segment_person_warning}")

                helmet = 0.53

                category = classes[idx].astype('int')
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                if person_warning > 0.15:
                    text_color = (76, 0, 255)
                #                     elif person_warning >= 0.15:
                #                         text_color = (25, 211, 249)
                else:
                    text_color = (166, 32, 27)

                # Рисуем boundbox
                cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), text_color, 2)  # box
                cv2.putText(img=input_image, text=f'Person : {int(confidence * 100)}%',
                            org=(xmin, ymin - 70), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)
                cv2.putText(img=input_image, text=f'Warning: {int(person_warning * 100)}%',
                            org=(xmin, ymin - 40), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)
                cv2.putText(img=input_image, text=f'Segment W: {int(segment_person_warning * 100)}%',
                            org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=text_color,
                            thickness=1)

                if is_segments:
                    # Рисуем сегментацию
                    segment_image = input_image.copy()
                    cv2.polylines(img=segment_image, pts=[segment], isClosed=True, color=(129, 176, 30), thickness=2)
                    cv2.fillPoly(segment_image, pts=[segment], color=(150, 190, 37))
                    input_image = cv2.addWeighted(segment_image, self.alpha, input_image, 1 - self.alpha, 0)
                    cv2.polylines(img=input_image, pts=[segment], isClosed=True, color=(129, 176, 30), thickness=2)

                # Выводим итоговое изображение
                # plt.figure(figsize = (20,22))
                # plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
                # Сохрнаяем файл со статистикаой по изображению
                info_predict = [file_name, confidence, person_warning, helmet]

                image_stat['count_persons'] += 1
                image_stat['person'] = max(image_stat['person'], confidence)
                image_stat['warning'] = max(image_stat['warning'], person_warning)
                image_stat['seg_warning'] = max(image_stat['seg_warning'], segment_person_warning)
                image_stat['helmet'] = max(image_stat['helmet'], helmet)

            # Сохрнаяем итоговое изображение
            output_filename = output_dir_zone + "/result_" + file_name
            cv2.imwrite(output_filename, input_image)
            self.result_df.loc[len(self.result_df.index)] = image_stat
            # self.result_df.loc[len(self.result_df.index)] = info_predict

            with open(output_dir_zone + "/result_" + file_name.split(".")[0] + ".txt", 'w') as txt_file:
                txt_stat = [str(image_stat[col_name]) for col_name in self.data_columns]
                txt_file.write("\t".join(txt_stat))
            return output_filename

    def detected_by_dir(self, input_dir: str, file_types: str, output_dir: str):
        self.load_photos(path_cameras=input_dir, file_types=file_types)
        for zone in tqdm(self.photos_by_zone.keys()):
            output_dir_zone = f"{output_dir}/{zone}"
            # Создаем выходную директорию для зоны если её нет
            os.makedirs(output_dir_zone, exist_ok=True)
            for file_photo in tqdm(self.photos_by_zone[zone]):
                self.model_predict(img_files=[file_photo], zone_name=zone, output_dir_zone=output_dir_zone)
        return self.result_df

    def detected_by_file(self, input_file, zone_name, output_dir: str):
        output_dir_zone = f"{output_dir}/{zone_name}"
        # Создаем выходную директорию для зоны если её нет
        os.makedirs(output_dir_zone, exist_ok=True)
        output_filename = self.model_predict(img_files=[input_file], zone_name=zone_name, output_dir_zone=output_dir_zone)
        return self.result_df, output_filename




# Обработчик команды start
def start(update, context):
    update.message.reply_text('Привет! Отправьте мне изображение.')


# Обработчик получения фото
def handle_photo(update, context):
    user_id = update.message.from_user.id
    photo_file = update.message.photo[-1].get_file()
    photo_path = f'{CAMERAS_PATH}/{user_id}.jpg'
    photo_file.download(photo_path)

    # Обработка изображения
    output_filename = process_image(photo_path, user_id)

    # Отправляем обработанное изображение обратно пользователю
    # output_image_path = glob.glob(f'{OUTPUT_PATH}/*/{user_id}.jpg')[0]  # Получаем путь к обработанному изображению
    context.bot.send_photo(chat_id=update.message.chat_id, photo=open(output_filename, 'rb'))


def get_zone_name_by_photo(photo_path:str):
    result = reader.readtext(photo_path)
    text_detected = " ".join([detection[1] for detection in result])

    zones = {
        'com3': 'Phl-com3-Shv2-9-K34',
        'K3-8': 'Php-Angc-K3-8',
        'K3-1': 'Php-Angc-K3-1-1',
        'Ctm-K': 'Php-Ctm-K-1-12-56',
        'Ctm-Shv1': 'Php-Ctm-Shv1-2-K3',
        'nta': 'Php-nta4-shv016309-k2-1-7-nta4',
        'pc': 'Pgp-lpc2-K-0-1-38',
        'com2': 'Pgp-com2-K-1-0-9-36',
        'comz': 'Pgp-com2-K-1-0-9-36',
        'K1-3-3-5': 'Spp-210-K1-3-3-5',
        'K1-3-3-6': 'Spp-210-K1-3-3-6',
        'DpR': 'DpR-Csp-uipv-ShV-V1',
        'Spp-K1': 'Spp-K1-1-2-6'
    }

    for key in zones:
        if key in text_detected:
            print(key, zones[key])
            return zones[key]
    return None

# Функция обработки изображения
def process_image(photo_path, user_id):

    zone_name = get_zone_name_by_photo(photo_path)
    if zone_name is None:
        update.message.reply_text('Не удалось определить камеру')

    # zone_name = 'DpR-Csp-uipv-ShV-V1'
    # Производим детекцию, результат сохраняется в OUTPUT_PATH
    result_df, output_filename = model.detected_by_file(input_file=photo_path, zone_name=zone_name, output_dir=OUTPUT_PATH)
    result_dict = result_df.to_dict('records')
    print(result_dict)
    print(output_filename)
    return output_filename


# Основная функция
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    # Пути к директориям
    DATASET_PATH = 'mini_train_dataset_train/'
    DANGER_ZONES_PATH = DATASET_PATH + 'danger_zones/'
    CAMERAS_PATH = DATASET_PATH + 'cameras/'
    OUTPUT_PATH = 'output/'

    # Создаем экземпляр класса детекции
    device = 'cpu'  # Или 'cuda', если у вас есть поддержка GPU
    # Загружаем модель
    model_seg = YOLO("yolov8x-seg.pt")

    model = ModelDetected(model=model_seg, device=device)
    # Загрузка опасных зон
    model.load_danger_zones(path_zones=DANGER_ZONES_PATH)

    reader = easyocr.Reader(['en'])  # Инициализация только один раз

    main()
