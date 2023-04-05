import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import mariadb
import sys
import random
import folium




# Load the trained model
model = keras.models.load_model('C:/Anaconda/titanic.h5')
# MariaDB 연결
try:
    conn = mariadb.connect(
        user="root",
        password="1q2w3e4r",
        host="127.0.0.1",
        port=3306,
        database="temp"
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Cursor 가져오기 
cur = conn.cursor()
query = "SELECT 해수면온도 from citytemp4 WHERE 도시이름=?"

# Create a Streamlit app
st.title('Titanic Survived:ship:')
st.subheader("해수면 온도 변화에 따른 해양사고 생존율 예측 모델")
tab1, tab2, tab3 = st.tabs(["User", "Data", "Visual"])
with tab1:
    #st.markdown(":blue[**객실등급**]")
    #st.markdown('Streamlit is **_really_ cool**.')
    #st.markdown(":green[$a*x - c + e(-x-b)$] is a Pythagorean identity. :pencil:")
    st.subheader("Generate User Data")
    pclass = st.selectbox(':blue[***객실등급***]',[1,2,3])
    col1, col2 = st.columns(2)

    with col1:
        sex_str = st.radio(
            ":blue[***성별***]",
            ('남성', '여성'))

        if sex_str == '남성':
            sex=1
            sex2=0
        else:
            sex=0
            sex2=1
       

    with col2:
        age = st.slider(':blue[***나이***]', 0, 80, 25)
        st.write(age, '세')
    
    col1, col2 = st.columns(2)
    with col1:
        page_names = ['도시 선택','직접 입력']
        page = st.radio(':blue[***온도 설정***]', page_names)
    with col2:
        if page=='도시 선택':
            selected_name = st.selectbox(':blue[***도시***]', ['Halifax','Boston','New York','Charleston','Miami','Santiago de Cuba','Kingston',
                'Lima','Montevideo','Santa Marta'])
            cur.execute(query,(selected_name,))
            resultset = cur.fetchone()
            result = float(resultset[0])
            st.write('인근 해수면 온도 :', result,'°C')
        else:
            result = st.slider(':blue[***해수면 온도***]', -2, 27, 5)
            st.write(result,'도')
        
            
    fare = [70 if pclass==1 else 40 if pclass==2 else 20][0]
    col1, col2 = st.columns(2)
    @st.cache_data
    def random_num():
            a = float(random.randrange(1,60))
            return a
    with col1:
        page_names2 = ['직접 입력','랜덤 시간']
        page2 = st.radio(':blue[***구출시간 설정***]', page_names2)
    with col2:
        if page2 =='직접 입력':
            survive_time= st.slider(':blue[***구출시간***]', 1, 120, 50)
            st.write(survive_time,'분')
        else:
            def load_data(checkbox):
                if checkbox:
                    st.caption('구출시간이 고정되었습니다.')
                    survive_time = random_num()
                    return survive_time
                else:
                    survive_time = float(random.randrange(1,60))
                    return survive_time
            checkbox = st.checkbox('구출시간 고정', value=True) 
            survive_time = load_data(checkbox) 
            st.write(survive_time,'분')

      
       
    #if temp<10:
    #    weight = 1
    #elif temp>=10:
    #    weight = 2/3
    #elif temp >=20:
    #    weight = 1/2     
    
    #weight =float(result*-(7/99)+85/99 ) # 기울기 적용 가중치값 선택
    a = -0.016304819228334756
    b = 2.629378578845353
    c = -0.4346498560433844
    weight =float(np.exp(-result-b)+a*result-c)
    
        
    temp_survive_time = survive_time * weight

    tester = [ float(pclass), float(age)  ,  float(sex)  ,  float(sex2)  , float(fare),  1.  ,  0.  ,  1.  ,  0.  ,
            0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  1.  ,  0.  ,
            0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
            1.  , float(temp_survive_time)  ]
    tester = np.array(tester)
    data = (tester.reshape(1,29))
    # Use the model to predict the digit
    prediction = model.predict(data)
    data = prediction[0][0]
    # Display the predicted digit
    x = round(data, 2)
    percent_x = round(x * 100, 0)
    st.subheader('Result')
    
    col1, col2, col3 = st.columns(3)
    #col1.metric("객실등급", ''+str(pclass)+'등급')
    if pclass == 1:
        col1.metric("객실등급", "1st")
    elif pclass == 2:
        col1.metric("객실등급", "2nd")
    elif pclass == 3:
        col1.metric("객실등급", "3rd")
    col2.metric("성별", str(sex_str))
    col3.metric("나이", ''+str(age)+'세')
    col4, col5, col6 = st.columns(3)
    col4.metric("요금", ''+str(fare)+'$')
    col5.metric("해수면 온도", ''+str(result)+'°C')
    col6.metric("구출시간", ''+str(survive_time)+'분')
    st.write('예측값 : ', data,'')
    st.subheader('생존 확률 : '+str(percent_x)+'%')
    st.caption('*DNN 모델 정확도 : 90 %') 

    # 연결 닫기 
    #conn.close ()
with tab2:
    # # 데이터셋 불러오기
    # file_path = ('C:/Anaconda/master.csv')

    # df = pd.read_csv(file_path, encoding='UTF-8')
    # df = df.drop(['Unnamed: 0'], axis=1)

    # 지도 시작 지점 지정(타이타닉 사고 지점)
    if page=='도시 선택':
        la =[]
        lo = []
        k = []
        query2 = "SELECT 위도,경도 from citytemp2 WHERE 도시이름=?"
        cur.execute(query2,(selected_name,))
        resultset2 = cur.fetchone()
        lat = float(resultset2[0])
        lot = float(resultset2[1])
        
        k = pd.DataFrame({"latitude": [lat], "longitude": [lot]})
        st.subheader('Map of Data ')
        st.map(k)
        st.write(k)
    if page =='직접 입력':
        # 데이터셋 불러오기
        file_path = ('C:/Anaconda/master.csv')

        df = pd.read_csv(file_path, encoding='UTF-8')
        df = df.drop(['Unnamed: 0'], axis=1)
        la = []
        lo = []
        for i in range(len(df)):
            a = df.loc[i]['위도']
            b = df.loc[i]['경도']
            la.append(a)
            lo.append(b)
            
            df2 = pd.DataFrame(list(zip(la,lo)), columns = (['latitude', 'longitude']))

          
        st.subheader('Map of Data ')
        st.map(df2)
    add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))
    
    # 생성한 USER 데이터 DB에 저장
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pclass INT,
            age INT,
            sex_str VARCHAR(255),
            fare INT,
            result FLOAT,
            survive_time INT,
            data FLOAT
        )
    ''')
    

    # 데이터프레임 만들기
    df5 = pd.DataFrame({"객실등급": [pclass], "나이": [age], "성별": [sex_str], "지불요금":[fare],"사고지온도":[result],"구출시간":[survive_time],"생존예측율":[data]})

    # 데이터프레임 출력
    st.subheader("Generated User Data")
    st.write(df5)
    if st.button("Save To DB! :pencil:"):
        # 데이터베이스에 데이터 저장
        fare = float(fare) # numpy.float32 -> float 변환
        result = float(result) # numpy.float32 -> float 변환
        survive_time = float(survive_time) # numpy.float32 -> float 변환
        data = str(data) # numpy.ndarray -> str 변환
        sql2 = "INSERT INTO users (pclass, age, sex_str,fare, result,survive_time,data) VALUES (%s, %s, %s,%s,%s,%s,%s)"
        val = (pclass, age, sex_str,fare, result,survive_time,data)
        cur.execute(sql2, val)
        conn.commit()
        st.success("저장되었습니다.")

    # 데이터베이스 연결 종료
    cur.close()
    conn.close()

    
    
with tab3:
    from PIL import Image
    image = Image.open('가중치그래프.jpg')

    st.image(image, caption='가중치그래프')
    st.markdown(":green[$a*x - c + e(-x-b)$] ")
    #st.bar_chart(data=df, x='위도', y='최고기온', width=0, height=0, use_container_width=True)
   
    # from streamlit_folium import st_folium
    # m = folium.Map(location=[41.46, -50.24 ], 
                   # zoom_start=3)
    # for i in range(len(df)):
        # # 마커 생성(위도, 경도)
        # marker = folium.Marker([df.loc[i]['위도'], df.loc[i]['경도']],
                               # popup = (df.loc[i]['도시이름'],df.loc[i]['최고기온'])  # 마커를 클릭했을 때 팝업
                              # )
        # marker.add_to(m)
    # m.add_child(folium.LatLngPopup())   # 지도를 클릭했을 때 위도, 경도 표시
    # st_data = st_folium(m, width=725)