import face_recognition
import pickle
import os
import numpy as np
import cv2

class FaceSystem:
    def __init__(self, storage_file='faces.pkl'):
        """
        Инициализация системы распознавания лиц.

        Args:
            storage_file (str): Путь к файлу, где хранятся данные о лицах.
        """
        self.storage_file = storage_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_data()

    def load_data(self):
        """
        Загружает известные лица и их имена из файла.
        """
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get("encodings", [])
                    self.known_face_names = data.get("names", [])
                print(f"Загружено {len(self.known_face_names)} лиц.")
            except Exception as e:
                print(f"Ошибка при загрузке данных: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            print("Файл с данными не найден. Будет создан новый.")

    def save_data(self):
        """
        Сохраняет текущие известные лица и их имена в файл.
        """
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)
            print("Данные успешно сохранены.")
        except Exception as e:
            print(f"Ошибка при сохранении данных: {e}")

    def detect_faces(self, frame, scale_factor=0.25):
        """
        Обнаруживает лица на кадре. Для ускорения кадр уменьшается.

        Args:
            frame (numpy.ndarray): Кадр с веб-камеры.
            scale_factor (float): Коэффициент уменьшения кадра (0.25 = 1/4 размера).

        Returns:
            list: Список кортежей (top, right, bottom, left) с координатами лиц (масштабированными обратно).
            list: Список кодировок (embeddings) найденных лиц.
        """
        # Уменьшаем размер кадра для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Конвертируем BGR (OpenCV) в RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Находим лица и их кодировки
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Масштабируем координаты обратно
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)
            scaled_locations.append((top, right, bottom, left))

        return scaled_locations, face_encodings

    def recognize_faces(self, face_encodings, tolerance=0.6):
        """
        Сравнивает найденные кодировки с базой известных лиц.

        Args:
            face_encodings (list): Список кодировок лиц с текущего кадра.
            tolerance (float): Порог совпадения (меньше = строже).

        Returns:
            list: Список имен (str). Если лицо не найдено, имя = "Unknown".
        """
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            # Если нашли совпадение
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
        return face_names

    def add_new_face(self, name, face_encoding):
        """
        Добавляет новое лицо в базу.

        Args:
            name (str): Имя человека.
            face_encoding (numpy.ndarray): Кодировка лица.
        """
        self.known_face_names.append(name)
        self.known_face_encodings.append(face_encoding)
        self.save_data()
