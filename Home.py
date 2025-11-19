from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# ====== ตั้งค่าหน้าเว็บ ======
st.set_page_config(page_title="โปรเจคการจำแนกข้อมูลดอกไม้", layout="wide")

# ====== แบนเนอร์หัวเรื่อง ======
st.markdown("""
<div style="
    background: linear-gradient(90deg,#EC7063,#F39C9C);
    padding:18px;
    border-radius:12px;
    border:1px solid #000000;
    text-align:center;
">
  <h1 style="color:white; margin:0;">โปรเจคการจำแนกข้อมูลดอกไม้</h1>
  <div style="color:rgba(255,255,255,0.9); margin-top:6px;">ใช้ เค-ไนเออร์เนสต์ (K-Nearest Neighbors)</div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# รูปผู้ทำโปรเจค
left, right = st.columns([1,4])
with left:
    st.image("./img/Nawapath.jpg", width=160)
with right:
    st.write("")  # ว่างไว้เพื่อให้ภาพชิดซ้าย

st.markdown("---")

# ====== ตัวอย่างรูปดอกไม้ (ตรงกับการแมปด้านล่าง) ======
st.subheader("ตัวอย่างดอกไม้ (ภาพอ้างอิง)")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h4 style='text-align:center; margin-bottom:4px;'>Versicolor</h4>", unsafe_allow_html=True)
    st.image("./img/iris1.jpg", use_column_width=True)
with col2:
    st.markdown("<h4 style='text-align:center; margin-bottom:4px;'>Virginica</h4>", unsafe_allow_html=True)
    st.image("./img/iris2.jpg", use_column_width=True)
with col3:
    st.markdown("<h4 style='text-align:center; margin-bottom:4px;'>Setosa</h4>", unsafe_allow_html=True)
    st.image("./img/iris3.jpg", use_column_width=True)

st.markdown("---")

# ====== กล่องสถิติข้อมูล ======
st.markdown("""
<div style="background-color:#EC7063;color:white;padding:12px;border-radius:10px;border:1px solid #000;text-align:center;">
  <h4 style="margin:4px;">สถิติข้อมูลดอกไม้</h4>
</div>
""", unsafe_allow_html=True)

# โหลดข้อมูล
dt = pd.read_csv("./data/iris.csv")
st.write("ตัวอย่างข้อมูล 10 แถวแรก:")
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

# ====== กล่องทำนายข้อมูล ======
st.markdown("""
<div style="background-color:#6BD5DA;padding:12px;border-radius:10px;border:1px solid #000;text-align:center;">
  <h4 style="margin:4px;">ทำนายข้อมูล</h4>
</div>
""", unsafe_allow_html=True)
st.markdown("")

# --- เอาค่า min/max จากข้อมูลจริงมาใช้ตั้งสไลเดอร์/number input ---
pt_len_min, pt_len_max = float(dt['petallength'].min()), float(dt['petallength'].max())
pt_wd_min, pt_wd_max  = float(dt['petalwidth'].min()),  float(dt['petalwidth'].max())
sp_len_min, sp_len_max = float(dt['sepallength'].min()), float(dt['sepallength'].max())
sp_wd_min, sp_wd_max  = float(dt['sepalwidth'].min()),  float(dt['sepalwidth'].max())

# จัดคอลัมน์ใส่ข้อมูลให้สวย
colA, colB = st.columns(2)
with colA:
    st.markdown("**ข้อมูลกลีบดอก (Petal)**")
    petal_length = st.slider(
        "ความยาวกลีบดอก (petallength)",
        min_value=pt_len_min, max_value=pt_len_max,
        value=(pt_len_min + pt_len_max) / 2, step=0.1
    )
    petal_width = st.slider(
        "ความกว้างกลีบดอก (petalwidth)",
        min_value=pt_wd_min, max_value=pt_wd_max,
        value=(pt_wd_min + pt_wd_max) / 2, step=0.1
    )
with colB:
    st.markdown("**ข้อมูลกลีบเลี้ยง (Sepal)**")
    sepal_length = st.number_input(
        "ความยาวกลีบเลี้ยง (sepallength)",
        min_value=sp_len_min, max_value=sp_len_max,
        value=(sp_len_min + sp_len_max) / 2, step=0.1
    )
    sepal_width = st.number_input(
        "ความกว้างกลีบเลี้ยง (sepalwidth)",
        min_value=sp_wd_min, max_value=sp_wd_max,
        value=(sp_wd_min + sp_wd_max) / 2, step=0.1
    )

st.markdown("")

# ====== ปุ่มทำนาย ======
if st.button("🔍 ทำนายผล"):
    # เตรียมข้อมูล (ใช้คอลัมน์จากไฟล์จริงเพื่อความถูกต้องของลำดับ)
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    # สร้างและฝึกโมเดล
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # สร้าง input ตามลำดับคอลัมน์จริงของ X
    # ตัวแปรที่มี: petal_length, petal_width, sepal_length, sepal_width
    # แต่ลำดับคอลัมน์ใน X อาจต่างกัน ดังนั้น map ตามชื่อคอลัมน์
    input_map = {
        'petallength': petal_length,
        'petalwidth': petal_width,
        'sepallength': sepal_length,
        'sepalwidth': sepal_width
    }

    ordered_input = [ input_map[col] for col in X.columns ]
    x_input = np.array([ordered_input])

    prediction = model.predict(x_input)[0]
    st.success(f"ผลการทำนาย: {prediction}")

    # ====== แมปผลการทำนายให้ตรงกับรูปข้างบน ======
    # กำหนดให้:
    # - 'Versicolor' -> ./img/iris1.jpg
    # - 'Virginica' (หรือสะกดผิดเป็น 'Verginiga') -> ./img/iris2.jpg
    # - 'Setosa' -> ./img/iris3.jpg
    pred_lower = str(prediction).lower()
    if 'versicolor' in pred_lower:
        st.image("./img/iris1.jpg", caption="Versicolor (รูปอ้างอิง)")
    elif 'virgin' in pred_lower or 'vergin' in pred_lower:  # รองรับ 'Virginica' หรือ 'Verginiga'
        st.image("./img/iris2.jpg", caption="Virginica (รูปอ้างอิง)")
    elif 'setosa' in pred_lower:
        st.image("./img/iris3.jpg", caption="Setosa (รูปอ้างอิง)")
    else:
        st.write("พบสายพันธุ์ที่ไม่คุ้นเคย: ", prediction)

else:
    st.info("ยังไม่ได้ทำการทำนายผล — กดปุ่ม 'ทำนายผล' เพื่อให้โมเดลทำนาย")
