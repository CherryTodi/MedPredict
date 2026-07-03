import streamlit as st
from streamlit_option_menu import option_menu 
import pickle
import warnings
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.preprocessing import StandardScaler

from codebase.dashboard_graphs import MaternalHealthDashboard


st.set_page_config(
    page_title="MedPredict ",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
/* header {visibility:hidden;} */

.stApp{
    background:#F6F9FC;
    color:#1E293B;
}

section[data-testid="stSidebar"]{
    background:#0F172A;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Titles */
h1,h2,h3,h4{
    color:#1E293B !important;
}

/* Paragraph */
p,li,label{
    color:#475569 !important;
}

/* Metric */
[data-testid="stMetricValue"]{
    color:#2563EB !important;
    font-weight:bold;
}

[data-testid="stMetricLabel"]{
    color:#64748B !important;
}

.stTextInput input{
    border-radius:12px;
    border:1px solid #CBD5E1;
}

.stButton>button{
    background:#2563EB;
    color:white !important;
    border-radius:10px;
    border:none;
    padding:10px 22px;
}

.stButton>button:hover{
    background:#1D4ED8;
    color:white !important;}
.card{
    background:white;
    border-radius:18px;
    padding:25px;
    box-shadow:0 8px 18px rgba(0,0,0,.08);
    margin-bottom:20px;
}

.hero{
    background:linear-gradient(135deg,#2563EB,#38BDF8);
    border-radius:22px;
    padding:50px;
    text-align:center;
    color:white;
    margin-bottom:30px;}

.hero h1,
.hero h2,
.hero h3,
.hero p{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)


maternal_model = pickle.load(open("model/finalized_maternal_model.sav",'rb')) #load the saved model from model folder
fetal_model = pickle.load(open("model/fetal_health_classifier.sav",'rb'))
scale_X = pickle.load(open('model/scaler_maternal_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    st.title("🩺 MedPredict")
    st.markdown(" Choose an option from the menu below to get started:")
    st.markdown("---")

    selected = option_menu('MedPredict',
                          ['Home',
                           'About us',
                           'Pregnancy Risk Prediction',
                           'Fetal Health Prediction',
                           'Dashboard',],
                         icons=["house-fill","info-circle-fill","activity","heart-pulse-fill","bar-chart-fill",],
                          default_index=0,) #up to here is sidebar

if selected == "Home":
    st.markdown("""
    <div style='
        background:linear-gradient(135deg,#2563EB,#0EA5E9);
        padding:45px;
        border-radius:20px;
        color:white;
        text-align:center; '>
    <h1 style='font-size:55px;'>🩺 MedPredict</h1>
    <h3>
    Smart Maternal & Fetal Health Prediction System
    </h3>
    <p style='font-size:18px;'>
    Early prediction of maternal and fetal health risks.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="color:#1e293b;">Welcome</h2>
    <p style="font-size:18px;color:#475569;line-height:1.8;">
MedPredict is an intelligent healthcare platform designed to assist in the early prediction of maternal and fetal health risks.
The application combines Machine Learning with interactive analytics to support informed clinical decision-making.</p>
""", unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)
    with col1:
      st.markdown("""
      <div class="card">
      <h2>🤰 Maternal Prediction</h2>
      <p>Predict maternal pregnancy risk using:</p>
      <ul>
      <li>Age</li>
      <li>Blood Pressure</li>
      <li>Blood Sugar</li>
      <li>Body Temperature</li>
      <li>Heart Rate</li>
      </ul>
      </div>
    """, unsafe_allow_html=True)

    with col2:
      st.markdown("""
      <div class="card">
      <h2>👶 Fetal Prediction</h2>
      <p>Predict fetal health status using:</p>
      <ul>
      <li>CTG Features</li>
      <li>Machine Learning</li>
      <li>Instant Results</li>
      <li>Analytics</li>
       </ul>
      </div>
    """, unsafe_allow_html=True)


    with col3:
      st.markdown("""
      <div class="card">
      <h2>📊 Analytics</h2>
      <p>Explore:</p>
      <ul>
      <li>Interactive Dashboard</li>
      <li>Risk Distribution</li>
      <li>Health Insights</li>
      <li>Visual Analytics</li>
      </ul>
      </div>
      """,unsafe_allow_html=True)

    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("Maternal Features", "5")
    b.metric("Fetal Features","21")
    c.metric("ML Models", "2") 
    d.metric("Prediction Time", "<1 sec")
    st.markdown("---")

    st.subheader("Why MedPredict?")
    left,right = st.columns([1.2,1])

    with left:   
        st.markdown("-> Early Risk Detection")
        st.markdown("-> Easy-to-use Interface")
        st.markdown("-> Interactive Health Analytics")
        st.markdown("-> Supports Maternal & Fetal Health Monitoring") 


    with right:
      st.image(
        "graphics/ChatGPT Image Jun 30, 2026, 08_58_27 PM.png",
        width="content" )


if (selected == 'About us'): 
    st.header("Welcome to MedPredict")
    st.write("At MedPredict, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
         "Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate "
         "predictions and proactive risk management.")
    
    col1, col2= st.columns(2)
    with col1:
        # Section 1: Pregnancy Risk Prediction
        st.header("1. Pregnancy Risk Prediction")
        st.markdown("Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, "
                "body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of "
                "potential risks during pregnancy.")
        # Add an image for Pregnancy Risk Prediction
        st.image("graphics/ChatGPT Image Jun 30, 2026, 08_51_03 PM.png", caption="Pregnancy Risk Prediction", width="content")
    with col2:
        # Section 2: Fetal Health Prediction
        st.header("2. Fetal Health Prediction")
        st.markdown("Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the "
                "health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, "
                "and genetic factors, we deliver insights into the well-being of the unborn child.")
        # Add an image for Fetal Health Prediction
        st.image("graphics/ChatGPT Image Jun 30, 2026, 08_58_27 PM.png", caption="Fetal Health Prediction", width="content")

    # Section 3: Dashboard
    st.header("3. Dashboard")
    st.markdown("Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
            "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
            "is designed for ease of use and accessibility.")
    
    # Closing note
    st.markdown("Thank you for choosing E-Doctor. We are committed to advancing healthcare through technology and predictive analytics. "
            "Feel free to explore our features and take advantage of the insights we provide.")

if (selected == 'Pregnancy Risk Prediction'):
    
    # page title
    st.header('Pregnancy Risk Prediction')
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)  #ek row mai 3 col chahiye
    
    with col1:
       age = st.number_input("Age (years)", value=35.0,key="ag")
        
    with col2:
        diastolicBP = st.number_input("Diastolic BP (mmHg)", value=60.0, key="bp")
    
    with col3:
       BS = st.number_input("Blood Sugar( mmol/L)",value=6.10, key="bs")
    
    with col1:
        bodyTemp = st.number_input("Body Temperature (°F)",value=98.0, key="temp")

    with col2:
  #all these col should be in same format as in datset so that thet give correct prediction and input match to col
        heartRate = st.number_input("Heart Rate (bpm)", value=80.0,key="hr")
    riskLevel=""  #when eneter the values in col there must be submit button
    predicted_risk = [0] 
    # creating a button for Prediction
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #predicted_risk = maternal_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])
                input_data = np.array([[age, diastolicBP, BS, bodyTemp, heartRate]]) #2d array
        
                input_scaled = scale_X.transform(input_data)
                predicted_risk = maternal_model.predict(input_scaled) #maternal model jo upar load kia h uske basis p eresult load kro
            # st
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.success("Low Risk")  #success gives green color
                st.info("""
### Healthy Pregnancy Recommendations
• Continue regular prenatal check-ups.

• Maintain a balanced and nutritious diet.

• Engage in light physical activity such as walking.

• Drink plenty of water and stay hydrated.

• Get adequate sleep and manage stress.

• Continue prescribed prenatal vitamins and supplements.
""")

            elif predicted_risk[0] == 1:
                st.warning("Medium Risk")
                st.warning("""
### Medical Recommendations
• Schedule more frequent prenatal visits.

• Monitor blood pressure and blood sugar regularly.

• Follow a healthy, low-sugar diet.

• Avoid excessive physical exertion.

• Stay hydrated and maintain proper rest.

• Consult your healthcare provider if symptoms worsen.
""")               
            else:
                st.error("High Risk")
                st.error("""
### Immediate Medical Attention Recommended

• Consult a gynecologist or obstetrician immediately.

• Attend all scheduled prenatal appointments.

• Monitor blood pressure and blood sugar daily.

• Follow all prescribed medications strictly.

• Avoid smoking, alcohol, and self-medication.

• Seek emergency medical care if severe pain, bleeding, dizziness, or reduced fetal movement occurs.
""")
                
    with col2:
        if st.button("Clear"): #ab dubara test krna hai toh pehle wala clear karna parega so clear button
            st.rerun()

if (selected == 'Fetal Health Prediction'):
    
    # page title
    st.header('Fetal Health Prediction')
    
    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    # getting the input data from the user

    st.subheader("Heart Rate Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BaselineValue = st.number_input("Baseline Value (bpm)",value=120.0)
        
    with col2:
        Accelerations = st.number_input("Accelerations", value=0.0)
    
    with col3:
        fetal_movement = st.number_input("Fetal Movement", value=0.0)

    st.subheader("Contractions & Decelerations")
    col1, col2, col3 = st.columns(3)
    with col1:
        uterine_contractions = st.number_input("Uterine Contractions", value=0.0)

    with col2:
        light_decelerations = st.number_input("Light Decelerations", value=0.0)
    
    with col3:
        severe_decelerations = st.number_input("Severe Decelerations", value=0.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        prolongued_decelerations = st.number_input("Prolonged Decelerations", value=0.0)

    st.subheader("Variability Parameters")    
    with col2:
        abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", value=0.0)
    
    with col3:
        mean_value_of_short_term_variability = st.number_input("Mean Short Term Variability", value=0.0)

    col1, col2 = st.columns(2)
    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of Time with ALTV", value=0.0)

    with col2:
        mean_value_of_long_term_variability = st.number_input("Mean Long Term Variability", value=0.0)

    st.subheader("Histogram Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        histogram_width = st.number_input("Histogram Width", value=70.0)

    with col2:
        histogram_min = st.number_input("Histogram Min", value=50.0)
        
    with col3:
        histogram_max = st.number_input("Histogram Max", value=180.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        histogram_number_of_peaks = st.number_input("Histogram Peaks", value=2.0)
    with col2:
        histogram_number_of_zeroes = st.number_input("Histogram Zeroes", value=0.0)

    with col3:
        histogram_mode = st.number_input("Histogram Mode", value=120.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        histogram_mean = st.number_input("Histogram Mean", value=120.0)
    
    with col2:
        histogram_median = st.text_input("Histogram Median", value=120.0)

    with col3:
        histogram_variance = st.number_input("Histogram Variance", value=20.0)

    with col1:
        histogram_tendency = st.number_input("Histogram Tendency", value=0.0)
    
    # creating a button for Prediction
    st.markdown('</br>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Predict Pregnancy Risk',width="content"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = fetal_model.predict([[BaselineValue, Accelerations, fetal_movement,
       uterine_contractions, light_decelerations, severe_decelerations,
       prolongued_decelerations, abnormal_short_term_variability,
       mean_value_of_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_width,
       histogram_min, histogram_max, histogram_number_of_peaks,
       histogram_number_of_zeroes, histogram_mode, histogram_mean,
       histogram_median, histogram_variance, histogram_tendency]])
            # st.subheader("Risk Level:")
            st.markdown('</br>', unsafe_allow_html=True)
            if predicted_risk[0] == 0:
                st.success("Result Comes to be Normal")
            elif predicted_risk[0] == 1:
                st.warning("Result  Comes to be  Suspect")
            else:
                st.error("Result  Comes to be  Pathological")
    with col2:
        if st.button("Clear"): 
            st.rerun()  #same procedure as we did for maternal

if selected == "Dashboard":

    dashboard = MaternalHealthDashboard()
    dashboard.create_dashboard()