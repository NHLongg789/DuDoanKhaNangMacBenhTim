import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Tải mô hình và scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# Tiêu đề trang web
st.title("Dự đoán nguy cơ mắc bệnh tim")

# Mô tả
st.write("Nhập thông tin dưới đây để dự đoán nguy cơ mắc bệnh tim.")

# Các ô nhập liệu (bao gồm tất cả 13 đặc trưng)
age = st.number_input("Tuổi", min_value=0, max_value=120, value=50)
sex = st.selectbox("Giới tính", options=[("Nam", 1), ("Nữ", 0)], format_func=lambda x: x[0])[1]
cp = st.selectbox("Loại đau ngực", options=[
    ("Typical Angina (Đau ngực nặng khi gắng sức (như chạy bộ), nghỉ thì hết.)", 1),
    ("Atypical Angina (Đau ngực không rõ ràng, có thể lan ra vai, cổ, không nhất thiết khi gắng sức.)", 2),
    ("Non-anginal Pain (Đau ngực không do tim, ví dụ đau cơ, dạ dày.)", 3),
    ("Asymptomatic (Không đau ngực gì cả.)", 4)
], format_func=lambda x: x[0])[1]
trestbps = st.number_input("Huyết áp nghỉ (mm Hg) (Huyết áp khi bạn nghỉ, thường 90-120.)", min_value=0, value=120)
chol = st.number_input("Cholesterol (mg/dl) (Là lượng chất béo trong máu. Quá cao dễ gây bệnh tim.)", min_value=0, value=200)
fbs = st.selectbox("Đường huyết lúc đói (> 120 mg/dl) (Là lượng đường trong máu khi chưa ăn. Quá cao có thể báo hiệu tiểu đường.)", options=[("Không", 0), ("Có", 1)], format_func=lambda x: x[0])[1]
restecg = st.selectbox("Kết quả điện tâm nghỉ (Đo nhịp tim khi bạn nằm yên. Giúp phát hiện bất thường về tim.)", options=[
    ("Bình thường (Tim đập đều, không có dấu hiệu bất thường.)", 0),
    ("Có bất thường ST-T (Có thể có thiếu máu cơ tim hoặc rối loạn nhịp.)", 1),
    ("Phì đại thất trái (Buồng tim trái to lên, thường do cao huyết áp kéo dài.)", 2)
], format_func=lambda x: x[0])[1]
thalach = st.number_input("Nhịp tim tối đa (Nhịp tim cao nhất tim bạn có thể đạt khi gắng sức. Dùng để xác định mức tập luyện an toàn, từ 60–100 bpm là bình thường.)", min_value=0, value=150)
exang = st.selectbox("Đau thắt ngực do tập thể dục", options=[("Không", 0), ("Có", 1)], format_func=lambda x: x[0])[1]
oldpeak = st.number_input("ST chênh xuống (Dấu hiệu có thể liên quan đến thiếu máu cơ tim, từ < 0.5 mm (hoặc < 0.05 mV): An toàn, bình thường.)", min_value=0.0, value=1.0)
slope = st.selectbox("Độ dốc ST (Hình dạng đoạn ST trên điện tâm đồ khi gắng sức. Nó cho biết tim có đang thiếu máu hay không.)", options=[
    ("Dốc lên (Bình thường, tim khỏe.)", 1),
    ("Phẳng (Có thể báo hiệu vấn đề nhẹ về tim.)", 2),
    ("Dốc xuống (Thường là dấu hiệu cảnh báo thiếu máu cơ tim (nguy hiểm hơn))", 3)
], format_func=lambda x: x[0])[1]
ca = st.number_input("Số mạch máu chính bị hẹp (0: tim mạch bình thường, >0: có hẹp mạch máu có thể ảnh hưởng đến lưu thông máu tim)", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia (Là một bệnh thiếu máu di truyền, do cơ thể tạo ra hemoglobin bất thường (chất vận chuyển oxy trong máu))", options=[
    ("Bình thường (Không mắc bệnh thiếu máu di truyền)", 3),
    ("Khuyết tật cố định (Có gen bệnh, thiếu máu bẩm sinh, không hồi phục.)", 6),
    ("Khuyết tật có thể đảo ngược (Có dấu hiệu bệnh nhưng có thể cải thiện hoặc hồi phục.)", 7)
], format_func=lambda x: x[0])[1]

# Nút dự đoán
if st.button("Dự đoán"):
    # Tạo mảng dữ liệu đầu vào (13 đặc trưng)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data = np.zeros(len(columns))
    input_data[columns.index('age')] = age
    input_data[columns.index('sex')] = sex
    input_data[columns.index('cp')] = cp
    input_data[columns.index('trestbps')] = trestbps
    input_data[columns.index('chol')] = chol
    input_data[columns.index('fbs')] = fbs
    input_data[columns.index('restecg')] = restecg
    input_data[columns.index('thalach')] = thalach
    input_data[columns.index('exang')] = exang
    input_data[columns.index('oldpeak')] = np.log1p(oldpeak)  # Biến đổi log cho oldpeak
    input_data[columns.index('slope')] = slope
    input_data[columns.index('ca')] = ca
    input_data[columns.index('thal')] = thal
    input_data = input_data.reshape(1, -1)

    # Chuẩn hóa dữ liệu
    input_data = scaler.transform(input_data)

    # Dự đoán
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]  # Lấy xác suất dự đoán

    # Hiển thị kết quả dự đoán
    if prediction[0] == 1:
        st.error("Có nguy cơ mắc bệnh tim!")
    else:
        st.success("Không có nguy cơ mắc bệnh tim!")

    # Vẽ biểu đồ xác suất
    st.subheader("Xác suất dự đoán")
    fig, ax = plt.subplots()
    labels = ['Không mắc bệnh', 'Mắc bệnh']
    probs = [probabilities[0], probabilities[1]]
    ax.bar(labels, probs, color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Xác suất')
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    st.pyplot(fig)