import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from face_system import FaceSystem

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Инициализация системы распознавания
        self.face_system = FaceSystem()

        # Инициализация камеры
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            return

        # Получаем размеры видео
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Виджет для отображения видео
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas_image_id = None

        # Панель управления
        self.control_frame = tk.Frame(window)
        self.control_frame.pack(pady=10)

        # Поле ввода имени
        tk.Label(self.control_frame, text="Имя:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(self.control_frame)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        # Кнопка сохранения
        self.btn_save = tk.Button(self.control_frame, text="Сохранить лицо", command=self.save_face)
        self.btn_save.pack(side=tk.LEFT)

        # Переменные для хранения текущего кадра и обнаруженных лиц
        self.current_frame = None
        self.current_face_locations = []
        self.current_face_encodings = []
        self.current_face_names = []

        # Запускаем цикл обновления
        self.delay = 15 # миллисекунды
        self.update()

        self.window.mainloop()

    def save_face(self):
        """Обработчик кнопки сохранения лица"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Внимание", "Введите имя!")
            return

        # Ищем, есть ли на экране неизвестные лица
        unknown_found = False
        target_encoding = None

        for encoding, recognized_name in zip(self.current_face_encodings, self.current_face_names):
            if recognized_name == "Unknown":
                unknown_found = True
                target_encoding = encoding
                break

        if unknown_found and target_encoding is not None:
            self.face_system.add_new_face(name, target_encoding)
            messagebox.showinfo("Успех", f"Лицо сохранено как {name}")
            self.name_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Внимание", "Не найдено неизвестных лиц для сохранения.")

    def update(self):
        """Обновление кадра"""
        ret, frame = self.vid.read()
        if ret:
            self.current_frame = frame
            # Обрабатываем кадр (можно делать это не каждый кадр для оптимизации, но здесь делаем каждый для плавности)
            # Для "легкого" режима, face_recognition вызываем например каждый 5-й кадр,
            # но здесь я реализовал внутри detect_faces уменьшение кадра, что уже ускоряет.

            # Для еще большей производительности можно вынести обработку в отдельный поток,
            # но чтобы не усложнять код, оставим последовательно.

            # Детекция и распознавание
            self.current_face_locations, self.current_face_encodings = self.face_system.detect_faces(frame)
            self.current_face_names = self.face_system.recognize_faces(self.current_face_encodings)

            # Рисуем рамки и имена
            for (top, right, bottom, left), name in zip(self.current_face_locations, self.current_face_names):
                # Рисуем рамку
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Рисуем плашку с именем
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Конвертируем изображение для Tkinter (BGR -> RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)

            # Обновляем канвас
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            else:
                self.canvas.itemconfig(self.canvas_image_id, image=photo)

            self.canvas.image = photo # Сохраняем ссылку, чтобы не удалил сборщик мусора

        self.window.after(self.delay, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root, "Распознавание лиц")
