import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
import base64
from io import StringIO

# ⚙️ Cấu hình trang
st.set_page_config(page_title="Phân Loại Bánh Ngọt", layout="wide")

# Ẩn cảnh báo TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Danh sách nhãn và cấu hình
class_names = ['donut', 'su kem', 'sừng bò', 'tart trứng']
IMG_SIZE = (150, 150)

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = []

# Hàm tạo link download
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ket_qua_du_doan.csv" class="download-button">📥 Tải kết quả về máy</a>'
    return href

@st.cache_resource
def create_and_load_model():
    try:
        # Xây dựng architecture của model
        input_layer = Input(shape=(150, 150, 3))
        base_model = tf.keras.applications.MobileNet(
            input_tensor=input_layer,
            include_top=False,
            weights=None
        )
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output_layer = Dense(4, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        
        try:
            model = load_model('MobileNet_RGB-2506.h5')
        except:
            model.load_weights('MobileNet_RGB-2506.h5')
        return model
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo model: {str(e)}")
        st.stop()

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def show_prediction_chart(preds, labels):
    # Vẽ biểu đồ cột ngang
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, preds[0], color='#FF6B6B')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Độ tin cậy')
    plt.tight_layout()
    st.pyplot(plt)

    # Tạo DataFrame và styling
    df = pd.DataFrame({
        'Loại bánh': labels,
        'Độ tin cậy': [f"{x*100:.2f}%" for x in preds[0]]
    }).reset_index(drop=True)

    # Styling cho bảng
    st.markdown("""
        <style>
        .full-width {
            width: 100% !important;
        }
        .stDataFrame {
            width: 100% !important;
        }
        .download-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 1rem 0;
            text-align: center;
        }
        .download-button:hover {
            background: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0;'>
            <h4 style='color: #333; margin-bottom: 1rem;'>📊 Bảng thống kê chi tiết</h4>
        </div>
    """, unsafe_allow_html=True)

    # Hiển thị bảng
    st.dataframe(
        df.style
        .set_properties(**{
            'background-color': '#f8f9fa',
            'color': '#333',
            'font-size': '16px',
            'padding': '12px',
            'text-align': 'center',
            'width': '100%'
        })
        .set_table_styles([
            {'selector': '',
             'props': [('width', '100%')]},
            {'selector': 'th',
             'props': [
                 ('background-color', '#FF6B6B'),
                 ('color', 'white'),
                 ('font-weight', 'bold'),
                 ('text-align', 'center'),
                 ('padding', '12px'),
                 ('font-size', '16px')
             ]},
            {'selector': 'tr:hover',
             'props': [('background-color', '#f0f0f0')]}
        ]),
        use_container_width=True
    )

    # Thêm nút download
    st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='background: linear-gradient(to right, #ff758c, #ff7eb3); 
         padding: 2rem; border-radius: 12px; text-align: center;'>
        <h1 style='color: white;'>🍰 Phân Loại Bánh Ngọt với MobileNet</h1>
        <p style='color: white;'>Tải lên ảnh bánh để phân loại tự động bằng AI</p>
    </div>
""", unsafe_allow_html=True)

# Thêm thông tin về các loại bánh
st.markdown("""
    <div style='background: linear-gradient(to right, #fff6e6, #ffe6e6); 
         padding: 1.5rem; border-radius: 12px; margin: 2rem 0;'>
        <h2 style='text-align: center; color: #333; margin-bottom: 1.5rem;'>🎂 Các loại bánh có thể phân loại</h2>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem;'>
            <div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #000; font-size: 1.3rem; margin: 0;'>🍩 Donut</h3>
            </div>
            <div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #000; font-size: 1.3rem; margin: 0;'>🍮 Su Kem</h3>
            </div>
            <div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #000; font-size: 1.3rem; margin: 0;'>🥐 Sừng Bò</h3>
            </div>
            <div style='background: white; padding: 1.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #000; font-size: 1.3rem; margin: 0;'>🥮 Tart Trứng</h3>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load model
model = create_and_load_model()

# Main content
uploaded_file = st.file_uploader("Chọn ảnh bánh (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    left_col, center_col, right_col = st.columns([1, 1.2, 1])

    # Cột trái - Lịch sử
    with left_col:
        st.markdown("### 📝 Lịch sử phân tích")
        history_placeholder = st.empty()

    # Cột giữa - Upload và phân tích
    with center_col:
        img = Image.open(uploaded_file)
        st.image(img, caption="📷 Ảnh đã tải lên", width=600)

        if st.button("🔍 Phân tích ảnh"):
            with st.spinner("⏳ Đang dự đoán..."):
                input_data = preprocess_image(img)
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]
                class_name = class_names[predicted_class]

            if confidence >= 0.98:
                st.success(f"🍩 {class_name} với độ tin cậy {confidence*100:.2f}%")
                # Cập nhật lịch sử
                st.session_state.history.append({
                    "filename": uploaded_file.name,
                    "class": class_name,
                    "confidence": f"{confidence*100:.2f}%"
                })
                # Hiển thị lại lịch sử
                with history_placeholder.container():
                    for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                        st.markdown(f"**{idx}.** `{item['filename']}` → **{item['class']}** ({item['confidence']})")
            else:
                st.error("🚫 Không thể nhận diện ảnh này với độ tin cậy đủ cao")

    # Cột phải - Biểu đồ
    with right_col:
        st.markdown("### 📊 Thống kê dự đoán")
        if 'prediction' in locals():
            show_prediction_chart(prediction, class_names)

else:
    st.info("📂 Vui lòng chọn ảnh để bắt đầu phân tích")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; font-size: 18px;'>
        🚀 Phát triển bởi <strong>Nhóm 14</strong> · Sử dụng Python · TensorFlow · Streamlit
    </div>
""", unsafe_allow_html=True)