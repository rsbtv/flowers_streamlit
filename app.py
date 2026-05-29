import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import requests
import io
import pandas as pd

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="wide")

API_URL = st.secrets["API_URL"] if "API_URL" in st.secrets else "https://your-api-url/predict"

st.title("🌸 Классификация цветов")
st.write("Загрузите изображение или нарисуйте его на холсте, затем отправьте в API для классификации.")

# -------- Sidebar --------
st.sidebar.header("Параметры")
img_size = st.sidebar.number_input("Размер изображения для предобработки", min_value=28, max_value=256, value=32, step=1)
invert_canvas = st.sidebar.checkbox("Инвертировать рисунок с холста", value=False)
normalize_mode = st.sidebar.selectbox("Нормализация", ["standard", "tanh"])

# -------- Tabs --------
tab1, tab2 = st.tabs(["Загрузка изображения", "Рисование на холсте"])

uploaded_image = None
canvas_image = None

with tab1:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)

with tab2:
    st.write("Нарисуйте изображение ниже:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1.0)",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        canvas_array = canvas_result.image_data[:, :, :3].astype("uint8")
        canvas_image = Image.fromarray(canvas_array)
        st.image(canvas_image, caption="Рисунок с холста", use_container_width=False)

def preprocess_pil_image(image: Image.Image, size: int, normalize: str, invert: bool = False):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (size, size))

    if invert:
        image = ImageOps.invert(image)

    arr = np.array(image).astype("float32")

    if normalize == "tanh":
        arr = (arr / 127.5) - 1.0
    else:
        arr = arr / 255.0

    return image, arr

def pil_to_bytes(image: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Предпросмотр перед отправкой")

    selected_source = st.radio(
        "Источник изображения",
        ["Загруженное изображение", "Рисунок с холста"],
        index=0
    )

    selected_image = None
    if selected_source == "Загруженное изображение" and uploaded_image is not None:
        selected_image = uploaded_image
    elif selected_source == "Рисунок с холста" and canvas_image is not None:
        selected_image = canvas_image

    processed_preview = None
    if selected_image is not None:
        processed_preview, _ = preprocess_pil_image(
            selected_image,
            size=img_size,
            normalize=normalize_mode,
            invert=invert_canvas if selected_source == "Рисунок с холста" else False
        )
        st.image(processed_preview, caption=f"Предобработанное изображение {img_size}x{img_size}", use_container_width=False)
    else:
        st.info("Сначала загрузите изображение или нарисуйте его на холсте.")

with col2:
    st.subheader("Результаты классификации")

    if st.button("Отправить на API", type="primary"):
        if selected_image is None:
            st.warning("Нет изображения для отправки.")
        else:
            try:
                processed_preview, _ = preprocess_pil_image(
                    selected_image,
                    size=img_size,
                    normalize=normalize_mode,
                    invert=invert_canvas if selected_source == "Рисунок с холста" else False
                )

                img_bytes = pil_to_bytes(processed_preview, fmt="PNG")

                files = {
                    "file": ("image.png", img_bytes, "image/png")
                }

                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code != 200:
                    st.error(f"Ошибка API: {response.status_code}")
                    st.text(response.text)
                else:
                    result = response.json()

                    predicted_class = result.get("predicted_class", "unknown")
                    predicted_index = result.get("predicted_index", -1)
                    confidence = result.get("confidence", None)
                    probabilities = result.get("probabilities", {})

                    st.success(f"Предсказанный класс: {predicted_class}")
                    st.write(f"Индекс класса: {predicted_index}")

                    if confidence is not None:
                        st.write(f"Уверенность: {confidence:.4f}")

                    if probabilities:
                        df = pd.DataFrame({
                            "class": list(probabilities.keys()),
                            "probability": list(probabilities.values())
                        }).sort_values("probability", ascending=False)

                        st.subheader("Распределение вероятностей")
                        st.bar_chart(df.set_index("class"))

                        st.dataframe(df, use_container_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения с API: {e}")
            except Exception as e:
                st.error(f"Ошибка обработки: {e}")