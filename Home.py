from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ======================
# ตั้งค่าหน้าเว็บ
# ======================
st.set_page_config(page_title="โปรเจคการจำแนกข้อมูลดอกไม้", layout="wide")

# ======================
# กำหนดให้เนื้อหาเว็บกว้าง 80% และอยู่ตรงกลาง
# ======================
st.markdown("""
<style>
    .main-container {
        max-width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)


# ======================
# แบนเนอร์หัวเรื่อง
# ======================
st.markdown("""
<div style="
    background: linear-gradient(90deg,#EC7063,#F39C9C);
    padding:20px;
    border-radius:12px;
    border:1px solid #000000;
    text-align:center;
">
  <h1 style="color:white; margin:0;">โปรเจคการจำแนกข้อมูลดอกไม้</h1>
  <p style="color:white; margin-top:6px;">Machine Learning ด้วยวิธี K-Nearest Neighbors</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.image("./img/Nawapath.jpg", width=200, caption="ผู้จัดทำโปรเจค")
st.markdown("---")


# ======================
# ตัวอย่างรูปดอกไม้ (ตรงกับผลการทำนาย)
# ======================
st.subheader("ตัวอย่างดอกไม้แต่ละชนิด")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h4 style='text-align:center;'>Versicolor</h4>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)

with col2:
    st.markdown("<h4 style='text-align:center;'>Virginica</h4>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)

with col3:
    st.markdown("<h4 style='text-align:center;'>Setosa</h4>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)

st.markdown("---")


# ======================
# สถิติข้อมูลดอกไม้
# ======================
st.markdown("""
<div style="background-color:#EC7063;color:white;padding:12px;
            border-radius:10px;border:1px solid #000;text-align:center;">
  <h4 style="margin:4px;">สถิติข้อมูลดอกไม้</h4>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# โหลดข้อมูล
dt = pd.read_csv("./data/iris.csv")
st.write("ตัวอย่างข้อมูล (10 แถวแรก):")
st.dataframe(dt.head(10))

# สรุปค่า
sum_df = pd.DataFrame({
    "feature": ["petallength","petalwidth","sepallength","sepalwidth"],
    "sum": [
        dt['petallength'].sum(),
        dt['petalwidth'].sum(),
        dt['sepallength'].sum(),
        dt['sepalwidth'].sum()
    ]
})
st.table(sum_df)

# ปุ่มแสดงกราฟ
if st.button("แสดงการจินตทัศน์ข้อมูล (Bar Chart)"):
    st.bar_chart(sum_df.set_index('feature'))
else:
    st.write("ยังไม่แสดงกราฟ")

st.markdown("---")


# ======================
# ทำนายข้อมูลดอกไม้
# ======================
st.markdown("""
<div style="background-color:#6BD5DA;padding:12px;border-radius:10px;
            border:1px solid #000;text-align:center;">
  <h4 style="margin:4px;">ทำนายข้อมูลดอกไม้</h4>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# เอาค่า min/max จาก dataset
pt_len_min, pt_len_max = float(dt['petallength'].min()), float(dt['petallength'].max())
pt_wd_min, pt_wd_max  = float(dt['petalwidth'].min()),  float(dt['petalwidth'].max())
sp_len_min, sp_len_max = float(dt['sepallength'].min()), float(dt['sepallength'].max())
sp_wd_min, sp_wd_max  = float(dt['sepalwidth'].min()),  float(dt['sepalwidth'].max())

# คอลัมน์รับค่า
colA, colB = st.columns(2)

with colA:
    st.markdown("**ข้อมูลกลีบดอก (Petal)**")
    petal_length = st.slider("petallength", min_value=pt_len_min, max_value=pt_len_max,
                             value=(pt_len_min + pt_len_max)/2, step=0.1)
    petal_width = st.slider("petalwidth", min_value=pt_wd_min, max_value=pt_wd_max,
                            value=(pt_wd_min + pt_wd_max)/2, step=0.1)

with colB:
    st.markdown("**ข้อมูลกลีบเลี้ยง (Sepal)**")
    sepal_length = st.number_input("sepallength", min_value=sp_len_min, max_value=sp_len_max,
                                   value=(sp_len_min + sp_len_max)/2, step=0.1)
    sepal_width = st.number_input("sepalwidth", min_value=sp_wd_min, max_value=sp_wd_max,
                                  value=(sp_wd_min + sp_wd_max)/2, step=0.1)

st.markdown("")


# ======================
# ปุ่มทำนาย
# ======================
if st.button("🔍 ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # จัดลำดับ input ตามคอลัมน์ใน CSV
    inputs = {
        'petallength': petal_length,
        'petalwidth': petal_width,
        'sepallength': sepal_length,
        'sepalwidth': sepal_width
    }
    ordered_input = [inputs[col] for col in X.columns]
    x_input = np.array([ordered_input])

    prediction = model.predict(x_input)[0]
    st.success(f"ผลการทำนาย: {prediction}")

    # แสดงภาพตามผลทำนาย
    pred = prediction.lower()

    if "versicolor" in pred:
        st.image("./img/iris1.jpg", caption="Versicolor")
    elif "virgin" in pred:   # รองรับ Virginica / Verginiga
        st.image("./img/iris2.jpg", caption="Virginica")
    elif "setosa" in pred:
        st.image("./img/iris3.jpg", caption="Setosa")
    else:
        st.warning("ไม่พบภาพตัวอย่างของสายพันธุ์นี้")

else:
    st.info("กรอกข้อมูลด้านบน แล้วกดปุ่ม 'ทำนายผล'")

st.markdown('</div>', unsafe_allow_html=True)