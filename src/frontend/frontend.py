import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import numpy as np

st.set_page_config(
    page_title="Vehicle Detection System",
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
st.sidebar.markdown("""
**Бэкенд:** FastAPI  
**Модели:** YOLOv8  
**API URL:** http://localhost:8000  
**Документация:** [/docs](http://localhost:8000/docs)
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

def call_detection_api(file, model_name, conf_threshold):
    """Вызов API для детекции"""
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        params = {
            'model_name': model_name,
            'conf_threshold': conf_threshold
        }        
        response = requests.post(
            "http://localhost:8000/detect/",
            files=files,
            params=params,
            timeout=30
        )        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            st.error(f"Сообщение: {response.text}")
            return None            
    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к серверу. Убедитесь, что бэкенд запущен.")
        return None
    except Exception as e:
        st.error(f"Ошибка при вызове API: {str(e)}")
        return None

def call_compare_api(file, conf_threshold):
    """Вызов API для сравнения моделей"""
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        params = {'conf_threshold': conf_threshold}        
        response = requests.post(
            "http://localhost:8000/compare/",
            files=files,
            params=params,
            timeout=60
        )        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API при сравнении: {response.status_code}")
            return None            
    except Exception as e:
        st.error(f"Ошибка при вызове API сравнения: {str(e)}")
        return None

# Обработка кнопки "Обнаружить"
if uploaded_file is not None and 'detect_btn' in locals() and detect_btn:
    if model_choice == "yolov8m (точная)":
        model_name = "yolov8m"
    else:
        model_name = "yolov8n"

    with st.spinner("Выполняется детекция транспортных средств..."):
        result = call_detection_api(
            uploaded_file, 
            model_name, 
            confidence_threshold
        )        
        if result:
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
            
            # Обработанное изображение
            if "processed_image" in result:
                st.subheader("Результат детекции")
                try:
                    img_data = result["processed_image"].split(",")[1]
                    img_bytes = base64.b64decode(img_data)
                    processed_img = Image.open(io.BytesIO(img_bytes))
                    st.image(processed_img, caption="Обнаруженные транспортные средства", use_column_width=True)
                except Exception as e:
                    st.warning(f"Не удалось отобразить обработанное изображение: {str(e)}")
            

# Обработка кнопки "Сравнить модели"
if uploaded_file is not None and 'compare_btn' in locals() and compare_btn:
    with st.spinner("Сравниваю производительность моделей..."):
        comparison = call_compare_api(uploaded_file, confidence_threshold)        
        if comparison:
            st.success("Сравнение завершено!") 
            st.subheader("Сравнение производительности моделей")   
            results = comparison.get("results", {})            
            if results:
                # Создаем колонки для сравнения
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
                        f"{model_result.get('processing_time', 0):.3f} с"
                    )
                    vehicle_counts = model_result.get("vehicle_counts", {})
                    if vehicle_counts:
                        st.write("**Обнаружено:**")
                        for v_type, count in vehicle_counts.items():
                            st.write(f"- {v_type}: {count}")
                
                # Модель 2: yolov8m
                with col2:
                    st.markdown("### YOLOv8m (точная)")
                    model_result = results.get("yolov8m", {})                    
                    st.metric(
                        "Транспортных средств",
                        model_result.get("total_vehicles", 0)
                    )                    
                    st.metric(
                        "Время обработки",
                        f"{model_result.get('processing_time', 0):.3f} с"
                    )
                    vehicle_counts = model_result.get("vehicle_counts", {})
                    if vehicle_counts:
                        st.write("**Обнаружено:**")
                        for v_type, count in vehicle_counts.items():
                            st.write(f"- {v_type}: {count}")

                
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
                comp_data = comparison.get("comparison", {})
                speed_ratio = comp_data.get("speed_ratio", 1)
                count_diff = comp_data.get("count_difference", 0)                
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
                

# Информация при первом запуске
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
    "Vehicle Detection System | FastAPI + YOLOv8 + Streamlit"
    "</div>",
    unsafe_allow_html=True
)