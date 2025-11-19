from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.header("โปรเจคการจำแนกข้อมูลดอกไม้")
st.image("./img/Nawapath.jpg")

#st.image("./pic/kairung.jpg")
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Versicolor")
    st.image("./img/iris1.jpg")

with col2:
    st.header("Verginiga")
    st.image("./img/iris2.jpg")

with col3:
    st.header("Setosa")
    st.image("./img/iris3.jpg")



html_7 = """
<div style="background-color:#EC7063;color:white;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลดอกไม้</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")


dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    #st.write(dt.head(10))
    st.bar_chart(dx2)
    st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")


html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")


if st.button("ทำนายผล"):
    #st.write("ทำนาย")
    dt = pd.read_csv("./data/iris.csv") 
    X = dt.drop('variety', axis=1)
    y = dt.variety   
    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)   

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]]) #ข้อมูลใหม่

    st.write(Knn_model.predict(x_input))
    out=Knn_model.predict(x_input)

    if out[0] == 'Setosa':
        st.image("./img/iris1.jpg")
    elif out[0] == 'Versicolor':       
        st.image("./img/iris2.jpg")
    else:
        st.image("./img/iris3.jpg")
else:
    st.write("ไม่ทำนาย")


    

from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np

# =============================
#   Header + รูปโปรเจค
# =============================
st.header("🌸 โปรเจคการจำแนกข้อมูลดอกไม้ (Iris Classification)")
st.image("./img/Nawapath.jpg", caption="By Nawapath", use_column_width=True)

st.markdown("---")

# =============================
#   แสดงรูปดอกไม้แบบสวยงาม
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Setosa")
    st.image("./img/iris3.jpg", use_column_width=True)

with col2:
    st.subheader("Versicolor")
    st.image("./img/iris1.jpg", use_column_width=True)

with col3:
    st.subheader("Virginica")
    st.image("./img/iris2.jpg", use_column_width=True)

st.markdown("---")


# =============================
#   สถิติข้อมูล
# =============================
html_title1 = """
<div style="
    background-color:#EC7063;
    color:white;
    padding:15px;
    border-radius:12px;
    border:2px solid black;
    text-align:center;
">
<h5>📊 สถิติข้อมูลดอกไม้</h5>
</div>
"""
st.markdown(html_title1, unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.write(dt.head(10))

# =============================
#   Summary Chart
# =============================
dt1 = dt['petallength'].sum()
dt2 = dt['petalwidth'].sum()
dt3 = dt['sepallength'].sum()
dt4 = dt['sepalwidth'].sum()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["PetalLength", "PetalWidth", "SepalLength", "SepalWidth"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.info("⚠️ ยังไม่แสดงข้อมูล")


st.markdown("---")


# =============================
#   ส่วนทำนายผล
# =============================
html_title2 = """
<div style="
    background-color:#6BD5DA;padding:15px;border-radius:12px;border:2px solid black;text-align:center;
">
<h5>🔮 ทำนายข้อมูล</h5>
</div>
"""
st.markdown(html_title2, unsafe_allow_html=True)

pt_len = st.slider("เลือกค่า Petal Length", 0.0, 10.0, 1.0)
pt_wd = st.slider("เลือกค่า Petal Width", 0.0, 5.0, 1.0)

sp_len = st.number_input("กรอกค่า Sepal Length", 0.0, 10.0, 1.0)
sp_wd = st.number_input("กรอกค่า Sepal Width", 0.0, 5.0, 1.0)

if st.button("ทำนายผล"):
    # โหลดข้อมูล
    dt = pd.read_csv("./data/iris.csv")
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    # โมเดล KNN
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # ข้อมูลใหม่
    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    out = model.predict(x_input)

    st.success(f"🌼 ผลการทำนาย: {out[0]}")

    # แสดงรูปตามผลลัพธ์
    if out[0] == 'Setosa':
        st.image("./img/iris3.jpg", caption="Setosa")
    elif out[0] == 'Versicolor':
        st.image("./img/iris1.jpg", caption="Versicolor")
    else:
        st.image("./img/iris2.jpg", caption="Virginica")
else:
    st.info("⚠️ ยังไม่เริ่มทำนาย")

