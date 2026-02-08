import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
from pathlib import Path
import gdown

MODEL_URLS = {
    "yolov8n": "https://drive.google.com/uc?id=1j1bkbPgBeUr0LCnE_-c_s9xIYfH0Riic",
    "yolov8m": "https://drive.google.com/uc?id=1w7e8v7rIzap65SSA1eHoRsKosmjHSliy",
}
MODEL_FILES = {
    "yolov8n": "yolov8nano.pt",
    "yolov8m": "yolov8m.pt",
}

st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide"
)

st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .vehicle-card {
        background-color: #EFF6FF;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Vehicle Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Система детекции транспортных средств с использованием YOLOv8</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models_dict = {}    
    for model_name in MODEL_URLS.keys():
        model_file = MODEL_FILES[model_name]
        model_url = MODEL_URLS[model_name]
        
        if not os.path.exists(model_file):
            try:
                gdown.download(model_url, model_file, quiet=False)
            except Exception as e:
                model_file = f"{model_name}.pt"
        else:
            st.sidebar.info(f"Модель {model_name} уже скачана, использую локальную копию")
        
        try:
            models_dict[model_name] = YOLO(model_file)
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки {model_name}: {e}")
            try:
                models_dict[model_name] = YOLO(f"{model_name}.pt")
                st.sidebar.info(f"Использую стандартную модель {model_name} как fallback")
            except:
                st.sidebar.error(f"Не удалось загрузить даже стандартную модель {model_name}")    
    return models_dict

with st.spinner("Загрузка моделей..."):
    models_dict = load_models()

st.sidebar.header("Настройки")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Выберите модель:",
    ["yolov8n (быстрая)", "yolov8m (точная)"],
    index=0
)

confidence_threshold = st.sidebar.slider(
    "Порог уверенности:",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Минимальный уровень уверенности для обнаружения объектов"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Информация о системе")
st.sidebar.markdown(f"""
**Загружено моделей:** {len(models_dict)}  
**Модели:** YOLOv8  
""")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Загрузите изображение для анализа",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Выберите изображение с транспортными средствами"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

with col2:
    st.subheader("Управление")    
    if uploaded_file is not None:
        col_btn1, col_btn2 = st.columns(2)        
        with col_btn1:
            detect_btn = st.button("Обнаружить", type="primary", use_container_width=True)        
        with col_btn2:
            compare_btn = st.button("Сравнить модели", use_container_width=True)        
        st.markdown("---")
        st.subheader("Информация о файле")
        st.write(f"**Имя файла:** {uploaded_file.name}")
        st.write(f"**Тип:** {uploaded_file.type}")
        st.write(f"**Размер:** {uploaded_file.size / 1024:.2f} KB")
    else:
        st.info("Загрузите изображение для начала анализа")

class VehicleDetector:
    VEHICLE_CLASSES = {
        0: "car",
        1: "van",
        2: "truck",
        3: "tricycle",
        4: "awning-tricycle",
        5: "bus",
        6: "motor"
    }
    
    def __init__(self, model_name: str = "yolov8n"):
        self.model_name = model_name
        self.model = models_dict.get(model_name)
        if self.model is None:
            raise ValueError(f"Модель {model_name} не найдена. Доступные модели: {list(models_dict.keys())}")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25):
        start_time = time.time()
        results = self.model.predict(image, conf=confidence_threshold, verbose=False)
        processing_time = time.time() - start_time
        
        vehicles = []
        total_vehicles = 0
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.VEHICLE_CLASSES.get(class_id, f"class_{class_id}")
                total_vehicles += 1
                
                vehicles.append({
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_normalized": [
                        float(x1 / image.shape[1]),
                        float(y1 / image.shape[0]),
                        float(x2 / image.shape[1]),
                        float(y2 / image.shape[0])
                    ]
                })
        
        vehicle_counts = {}
        for vehicle in vehicles:
            vehicle_type = vehicle["class"]
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        
        return {
            "model": self.model_name,
            "total_vehicles": total_vehicles,
            "vehicle_counts": vehicle_counts,
            "vehicles": vehicles,
            "processing_time": round(processing_time, 3),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            },
            "results": results
        }

if uploaded_file is not None and detect_btn:
    model_name = "yolov8m" if "yolov8m" in model_choice.lower() else "yolov8n"
    
    with st.spinner(f"Выполняется детекция с помощью {model_name}..."):
        try:
            image_pil = Image.open(uploaded_file)
            image_pil = image_pil.convert('RGB')
            img_array = np.array(image_pil)            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            detector = VehicleDetector(model_name)
            result = detector.detect(img_array, confidence_threshold=confidence_threshold)
            
            result["filename"] = uploaded_file.name
            result["file_size"] = uploaded_file.size
            result["conf_threshold"] = confidence_threshold
            
            if "processed_image" not in result:
                output_image = img_array.copy()
                for vehicle in result.get("vehicles", []):
                    x1, y1, x2, y2 = vehicle["bbox"]
                    label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"
                    
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(output_image, label, (x1, y1-10), font, 0.5, (0, 255, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', output_image)
                img_str = base64.b64encode(buffer).decode('utf-8')
                result["processed_image"] = f"data:image/jpeg;base64,{img_str}"
            
            st.success("Детекция завершена успешно!")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Транспортных средств",
                    result.get("total_vehicles", 0)
                )            
            with col2:
                st.metric(
                    "Время обработки",
                    f"{result.get('processing_time', 0):.3f} с"
                )            
            with col3:
                st.metric(
                    "Модель",
                    "yolov8n" if result.get("model") == "yolov8n" else "yolov8m"
                )            
            with col4:
                width = result.get("image_size", {}).get("width", 0)
                height = result.get("image_size", {}).get("height", 0)
                st.metric(
                    "Размер изображения",
                    f"{width}×{height}"
                )
            
            st.subheader("Статистика по типам транспорта")
            vehicle_counts = result.get("vehicle_counts", {})            
            if vehicle_counts:
                cols = st.columns(len(vehicle_counts))
                for idx, (vehicle_type, count) in enumerate(vehicle_counts.items()):
                    with cols[idx % len(cols)]:
                        st.markdown(f'<div class="metric-card">{vehicle_type}<br><h3>{count}</h3></div>', 
                                  unsafe_allow_html=True)
            else:
                st.info("Транспортные средства не обнаружены")
            
            if "processed_image" in result:
                st.subheader("Результат детекции")
                try:
                    img_data = result["processed_image"].split(",")[1]
                    img_bytes = base64.b64decode(img_data)
                    processed_img = Image.open(io.BytesIO(img_bytes))
                    st.image(processed_img, caption="Обнаруженные транспортные средства", use_column_width=True)
                except Exception as e:
                    st.warning(f"Не удалось отобразить обработанное изображение: {str(e)}")
                    
        except Exception as e:
            st.error(f"Ошибка при обработке изображения: {str(e)}")

if uploaded_file is not None and compare_btn:
    with st.spinner("Сравниваю производительность моделей..."):
        try:
            image_pil = Image.open(uploaded_file)
            image_pil = image_pil.convert('RGB')
            img_array = np.array(image_pil)            
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = {}        
            for model_name in ["yolov8n", "yolov8m"]:
                start_time = time.time()
                detector = VehicleDetector(model_name)
                model_result = detector.detect(img_array, confidence_threshold=confidence_threshold)
                processing_time = time.time() - start_time
                
                model_result["total_processing_time"] = round(processing_time, 3)
                
                output_image = img_array.copy()
                for vehicle in model_result.get("vehicles", []):
                    x1, y1, x2, y2 = vehicle["bbox"]
                    label = f"{vehicle['class']}: {vehicle['confidence']:.2f}"
                    
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(output_image, label, (x1, y1-10), font, 0.5, (0, 255, 0), 2)
                
                _, buffer = cv2.imencode('.jpg', output_image)
                img_str = base64.b64encode(buffer).decode('utf-8')
                model_result["processed_image"] = f"data:image/jpeg;base64,{img_str}"
                
                results[model_name] = model_result
            
            st.success("Сравнение завершено!")
            st.subheader("Сравнение производительности моделей")
            
            if results:
                col1, col2 = st.columns(2)                
                with col1:
                    st.markdown("### YOLOv8n (быстрая)")
                    model_result = results.get("yolov8n", {})
                    st.metric(
                        "Транспортных средств",
                        model_result.get("total_vehicles", 0)
                    )
                    st.metric(
                        "Время обработки",
                        f"{model_result.get('total_processing_time', 0):.3f} с"
                    )
                    vehicle_counts = model_result.get("vehicle_counts", {})
                    if vehicle_counts:
                        st.write("**Обнаружено:**")
                        for v_type, count in vehicle_counts.items():
                            st.write(f"- {v_type}: {count}")                
                with col2:
                    st.markdown("### YOLOv8m (точная)")
                    model_result = results.get("yolov8m", {})
                    st.metric(
                        "Транспортных средств",
                        model_result.get("total_vehicles", 0)
                    )
                    st.metric(
                        "Время обработки",
                        f"{model_result.get('total_processing_time', 0):.3f} с"
                    )
                    vehicle_counts = model_result.get("vehicle_counts", {})
                    if vehicle_counts:
                        st.write("**Обнаружено:**")
                        for v_type, count in vehicle_counts.items():
                            st.write(f"- {v_type}: {count}")
                
                # Визуальное сравнение
                st.markdown("---")
                st.subheader("Визуальное сравнение результатов")
                img_col1, img_col2 = st.columns(2)                
                with img_col1:
                    st.markdown("### YOLOv8n (быстрая)")
                    model_result = results.get("yolov8n", {})
                    if "processed_image" in model_result:
                        try:
                            img_data = model_result["processed_image"].split(",")[1]
                            img_bytes = base64.b64decode(img_data)
                            processed_img_n = Image.open(io.BytesIO(img_bytes))
                            st.image(processed_img_n, caption="Результат YOLOv8n", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Не удалось отобразить изображение YOLOv8n: {str(e)}")
                    else:
                        st.info("Обработанное изображение не получено для YOLOv8n")
                
                with img_col2:
                    st.markdown("### YOLOv8m (точная)")
                    model_result = results.get("yolov8m", {})
                    if "processed_image" in model_result:
                        try:
                            img_data = model_result["processed_image"].split(",")[1]
                            img_bytes = base64.b64decode(img_data)
                            processed_img_m = Image.open(io.BytesIO(img_bytes))
                            st.image(processed_img_m, caption="Результат YOLOv8m", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Не удалось отобразить изображение YOLOv8m: {str(e)}")
                    else:
                        st.info("Обработанное изображение не получено для YOLOv8m")
                
                st.markdown("---")
                st.subheader("Сравнительный анализ")                
                speed_ratio = round(
                    results.get("yolov8m", {}).get("total_processing_time", 0) / 
                    max(results.get("yolov8n", {}).get("total_processing_time", 0.001), 0.001), 
                    2
                )
                count_diff = results.get("yolov8m", {}).get("total_vehicles", 0) - results.get("yolov8n", {}).get("total_vehicles", 0)
                
                col1, col2 = st.columns(2)                
                with col1:
                    if speed_ratio > 1:
                        st.info(f"YOLOv8m медленнее в {speed_ratio:.2f} раз")
                    else:
                        st.info("Модели работают с одинаковой скоростью")                
                with col2:
                    if count_diff > 0:
                        st.success(f"YOLOv8m обнаружила на {count_diff} объектов больше")
                    elif count_diff < 0:
                        st.warning(f"YOLOv8n обнаружила на {-count_diff} объектов больше")
                    else:
                        st.info("Модели обнаружили одинаковое количество объектов")                        
        except Exception as e:
            st.error(f"Ошибка при сравнении моделей: {str(e)}")


if uploaded_file is None:
    st.markdown("---")
    st.info("""
    ### Инструкция по использованию:
    1. **Загрузите изображение** в формате JPG, PNG или BMP
    2. **Выберите модель** в боковой панели:
        - **YOLOv8n**: Быстрая модель, оптимальная для реального времени
        - **YOLOv8m**: Точная модель, обеспечивающая лучшее качество детекции
    3. **Настройте порог уверенности** (рекомендуется 0.25)
    4. **Нажмите кнопку "Обнаружить"** для выполнения детекции
    5. Для сравнения производительности моделей используйте **"Сравнить модели"**
    """)


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "Vehicle Detection System | YOLOv8 + Streamlit | Версия: Облачная"
    "</div>",
    unsafe_allow_html=True

)


