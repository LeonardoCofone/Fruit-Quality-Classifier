import os
import sys
import time
import threading
import io
import numpy as np
from PIL import Image as PILImage, UnidentifiedImageError, ImageOps
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
import joblib
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.utils import platform
from kivy.clock import Clock

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path(os.path.join('models', 'FruitQuality.tflite'))
ENCODER_PATH = resource_path(os.path.join('models', 'label_encoder.pkl'))
THRESHOLD_PATH = resource_path(os.path.join('models', 'thresholds.pkl'))

interpreter = None
le = None
class_names = []
best_thresholds = None

FRUIT_CONFIDENCE_THRESHOLD = 0.95

FRUIT_NAMES = {
    'apple': 'Apple', 'banana': 'Banana', 'pepper': 'Pepper', 'carrot': 'Carrot',
    'cucumber': 'Cucumber', 'grape': 'Grape', 'guava': 'Guava', 'jujube': 'Jujube',
    'mango': 'Mango', 'orange': 'Orange', 'pomegranate': 'Pomegranate', 'potato': 'Potato',
    'strawberry': 'Strawberry', 'tomato': 'Tomato'
}

if platform in ('android', 'ios'):
    Window.clearcolor = (0.7, 0.9, 1, 1)
else:
    Window.size = (700, 650)
    Window.clearcolor = (0.7, 0.9, 1, 1)
    Window.set_icon(resource_path('assets/LOGO.png'))


def predict_image(image_path):
    global interpreter, le, class_names, best_thresholds

    if interpreter is None or le is None or best_thresholds is None:
        return "Error", "Error", 0.0

    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        with open(image_path, 'rb') as f:
            img_data = f.read()

        img = PILImage.open(io.BytesIO(img_data))
        img = ImageOps.exif_transpose(img).convert('RGB')

        MAX_DIM = 1000
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM), PILImage.LANCZOS)


        img = img.resize((128, 128), PILImage.LANCZOS)
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array.astype('float32'))
        interpreter.invoke()
        raw_predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        label_scores = dict(zip(class_names, raw_predictions))

        fruit_scores = {}
        for label, score in label_scores.items():
            fruit = label.split('_')[0]
            fruit_scores[fruit] = max(fruit_scores.get(fruit, 0), score)

        best_fruit = max(fruit_scores, key=fruit_scores.get)
        best_fruit_conf = fruit_scores[best_fruit]

        fresh_label = f"{best_fruit}_fresh"
        rotten_label = f"{best_fruit}_rotten"

        fresh_index = np.where(class_names == fresh_label)[0][0]
        rotten_index = np.where(class_names == rotten_label)[0][0]

        fresh_conf = raw_predictions[fresh_index]
        rotten_conf = raw_predictions[rotten_index]

        fresh_thresh = best_thresholds[fresh_index]
        rotten_thresh = best_thresholds[rotten_index]

        if best_fruit_conf < FRUIT_CONFIDENCE_THRESHOLD:
            if fresh_conf >= 0.5:
                return "Unknown", "fresh", fresh_conf
            elif rotten_conf >= 0.5:
                return "Unknown", "rotten", rotten_conf
            else:
                return "Unknown", "doubt", max(fresh_conf, rotten_conf)
        else:
            if fresh_conf >= fresh_thresh:
                return best_fruit, "fresh", fresh_conf
            elif rotten_conf >= rotten_thresh:
                return best_fruit, "rotten", rotten_conf
            else:
                return best_fruit, "doubt", max(fresh_conf, rotten_conf)


    except Exception as e:
        return "Unknown", "Unknown", 0.0


class LoadingScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)

        self.label_top = Label(
            text="Loading AI model...\nPlease wait a moment",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_top.bind(size=self.label_top.setter('text_size'))
        self.add_widget(self.label_top)

        self.loading_image = KivyImage(
            source=resource_path('assets/LOGO.png'),
            allow_stretch=False,
            keep_ratio=False,
            size_hint=(1, 0.6)
        )
        self.add_widget(self.loading_image)

        self.label_bottom = Label(
            text="Powered by Leonardo Cofone",
            font_size='20sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.2),
            halign='center',
            valign='middle'
        )
        self.label_bottom.bind(size=self.label_bottom.setter('text_size'))
        self.add_widget(self.label_bottom)

class FruitQualityApp(App):
    def build(self):
        self.title = "Food Quality - By Leonardo Cofone"
        self.root_layout = BoxLayout()
        self.loading_screen = LoadingScreen()
        self.root_layout.add_widget(self.loading_screen)

        Clock.schedule_once(lambda dt: self.load_resources(), 1)
        return self.root_layout

    def load_resources(self):
        def load():
            global interpreter, le, class_names, best_thresholds
            try:
                interpreter = tf.lite.Interpreter(model_path=MODEL_PATH )
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
            except Exception as e:
                interpreter = None

            try:
                le = joblib.load(ENCODER_PATH)
                class_names = le.classes_
            except Exception as e:
                le = None
                class_names = []

            try:
                thresholds_data = joblib.load(THRESHOLD_PATH)
                best_thresholds = thresholds_data.get('thresholds', np.array([0.5]*len(class_names)))
            except Exception as e:
                best_thresholds = np.array([0.5]*len(class_names))
            

            Clock.schedule_once(lambda dt: self.show_main_screen())

        threading.Thread(target=load).start()

    def show_main_screen(self):
        main_layout = BoxLayout(orientation='vertical', padding=15, spacing=10)

        self.title_label = Label(
            text=" Fruit and Vegetables Quality AI ",
            font_size='24sp',
            size_hint=(1, 0.1),
            color=(1, 1, 1, 1),
            bold=True,
            halign='center',
            valign='middle'
        )
        self.title_label.bind(size=self.title_label.setter('text_size'))
        with self.title_label.canvas.before:
            Color(0.02, 0.53, 0.82, 1)
            self.rect = Rectangle(size=self.title_label.size, pos=self.title_label.pos)
            self.title_label.bind(pos=self.update_rect, size=self.update_rect)
        main_layout.add_widget(self.title_label)

        self.image_panel = KivyImage(size_hint=(1, 0.6), allow_stretch=True, keep_ratio=True)
        main_layout.add_widget(self.image_panel)

        self.result_label = Label(
            text="Upload an image for prediction",
            font_size='20sp',
            size_hint=(1, 0.15),
            color=(0, 0.3, 0.25, 1),
            halign='center',
            valign='middle'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        main_layout.add_widget(self.result_label)

        button_layout = BoxLayout(size_hint=(1, 0.15), spacing=15)
        self.load_button = Button(
            text="Upload an image",
            background_color=(0.015, 0.62, 0.9, 1),
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_release=self.open_file_chooser
        )
        button_layout.add_widget(self.load_button)

        self.reset_button = Button(
            text="Reset",
            background_color=(0.015, 0.62, 0.9, 1),
            color=(1, 1, 1, 1),
            font_size='16sp',
            on_release=self.reset_app,
        )
        button_layout.add_widget(self.reset_button)

        main_layout.add_widget(button_layout)

        self.root_layout.clear_widgets()
        self.root_layout.add_widget(main_layout)
        self.main_layout = main_layout

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def open_file_chooser(self, instance):
        if platform in ('android', 'ios'):
            initial_path = '/'
        else:
            home = os.path.expanduser("~")
            downloads = os.path.join(home, 'Downloads')
            pictures = os.path.join(home, 'Pictures')

            if os.path.exists(pictures):
                initial_path = pictures
            elif os.path.exists(downloads):
                initial_path = downloads
            else:
                initial_path = home

        content = FileChooserIconView(path=initial_path, filters=['*.jpg', '*.jpeg', '*.png', '*.bmp'])
        popup = Popup(
            title="Select a picture of fruit or vegetables",
            content=content,
            size_hint=(0.9, 0.9)
        )

        def selected_callback(*args):
            if content.selection:
                selected = content.selection[0]
                self.load_image_and_predict(selected)
                popup.dismiss()

        content.bind(on_submit=selected_callback)
        popup.open()

    def load_image_and_predict(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                img_data = f.read()

            pil_image = PILImage.open(io.BytesIO(img_data))
            pil_image.load()

            pil_image = ImageOps.exif_transpose(pil_image).convert('RGB')

            MAX_DIM = 224
            if max(pil_image.size) > MAX_DIM:
                pil_image.thumbnail((MAX_DIM, MAX_DIM), PILImage.LANCZOS)

            display_img = pil_image.copy()
            display_img.thumbnail((350, 350), PILImage.LANCZOS)

            data = io.BytesIO()
            display_img.save(data, format='png')
            data.seek(0)

            try:
                self.image_panel.texture = self.load_texture(data)
            except Exception:
                self.result_label.text = "Error loading image"


        except UnidentifiedImageError:
            self.result_label.text = "Image not recognized or format not supported.\n Accepted formats: \".jpg\",\".jpeg\", \".png\", \".bmp\""
            self.result_label.color = (0.7, 0, 0, 1)
            return

        except Exception as e:
            self.result_label.text = f"Image loading error: {e}"
            self.result_label.color = (0.7, 0, 0, 1)
            return

        fruit_type, status, confidence = predict_image(file_path)

        if fruit_type.lower() != "unknown" and confidence >= FRUIT_CONFIDENCE_THRESHOLD:
            display_fruit = FRUIT_NAMES.get(fruit_type, fruit_type.capitalize())
        else:
            display_fruit = "Product"

        if status == "fresh":
            result_text = (f"{display_fruit} - Status: FRESH!\nConfidence: {confidence:.2%}" 
                        if display_fruit else f"Status: FRESH!\nConfidence: {confidence:.2%}")
            self.result_label.color = (0.16, 0.65, 0.27, 1)
        elif status == "rotten":
            result_text = (f"{display_fruit} - Status: ROTTEN!\nConfidence: {confidence:.2%}" 
                        if display_fruit else f"Status: ROTTEN!\nConfidence: {confidence:.2%}")
            self.result_label.color = (0.86, 0.21, 0.27, 1)
        elif status == "Unknown":
            result_text = "Unable to identify the product type or status with confidence."
            self.result_label.color = (0.5, 0.5, 0.5, 1)
        else:
            result_text = (f"{display_fruit} - Status: {status.upper()}" if display_fruit else f"Status: {status.upper()}")
            self.result_label.color = (1, 0.76, 0, 1)

        self.result_label.text = result_text
        self.reset_button.disabled = False


    def load_texture(self, data):
        from kivy.core.image import Image as CoreImage
        return CoreImage(data, ext="png").texture

    def reset_app(self, instance):
        self.image_panel.texture = None
        self.result_label.text = "Upload an image for prediction"
        self.result_label.color = (0.2, 0.2, 0.2, 1)


if __name__ == '__main__':
    FruitQualityApp().run()
