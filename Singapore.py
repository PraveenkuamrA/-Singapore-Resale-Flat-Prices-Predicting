import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide")
st.title('Singapore  Resale Flat Prices Predicting')

 
st.subheader('Domain: Real Estate')
st.caption(''':orange[The objective of this web application is to predicts the resale prices of flats in Singapore
            and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.]''')


town_option=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN', 'LIM CHU KANG']

flat_type_option=['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE',
       'MULTI-GENERATION', '1 ROOM', 'MULTI GENERATION']

storey_range_option = ['06 TO 10', '01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25',
       '26 TO 30', '36 TO 40', '31 TO 35', '04 TO 06', '01 TO 03',
       '07 TO 09', '10 TO 12', '13 TO 15', '19 TO 21', '22 TO 24',
       '16 TO 18', '25 TO 27', '28 TO 30', '37 TO 39', '34 TO 36',
       '31 TO 33', '40 TO 42', '49 TO 51', '46 TO 48', '43 TO 45']

flat_model_option = ['Improved-Maisonette', '3Gen', 'STANDARD', 'Model A-Maisonette',
       'SIMPLIFIED', 'Premium Apartment Loft', 'Premium Maisonette',
       'APARTMENT', 'Adjoined flat', 'MULTI GENERATION', 'Model A2',
       'TERRACE', 'DBSS', 'Type S1', 'Type S2', '2-ROOM', 'IMPROVED',
       'NEW GENERATION', 'MODEL A', 'MAISONETTE', 'PREMIUM APARTMENT']

with st.form("my_form"):
    col1, col2, col3 = st.columns([5, 2, 5])

    with col1: 
        st.write(
        f'<h5 style="color:#ee4647;">NOTE : Min & Max given for reference, you can enter any value</h5>',
        unsafe_allow_html=True)

        Month=st.text_input("Enter The Month(Min:1 & Max:12)")
        Year = st.text_input('Enter The Year')
        Floor_area_sqm = st.text_input("Enter the Square Feet(Min:28 & Max:310)")
        Lease_commence_date=st.text_input('Enter The Lease Commence Year')

    with col3: 
        st.write(
        f'<h5 style="color:#ee4647;">NOTE : ML Modelling Done By Decision Tree Regressor ( R^2 Score: 0.9109280505043144)</h5>',
        unsafe_allow_html=True)
        
        Town = st.selectbox("Town",town_option,key=1)
        Flat_model = st.selectbox('Flat_model',flat_model_option,key=2)
        Flat_type = st.selectbox("Flat_type", flat_type_option, key=3)
        Storey_range = st.selectbox('Storey_range',storey_range_option,key=4)

        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #004aad;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)

if submit_button: 

    with open(r'https://github.com/PraveenkuamrA/-Singapore-Resale-Flat-Prices-Predicting/blob/main/preprocessor11.pkl', 'rb') as file:
        guvi = pickle.load(file)

    with open(r"C:\Users\user\Downloads\preprocessor11.pkl", 'rb') as f:
        preprocessor = pickle.load(f)

    user_input=pd.DataFrame({
            'month':[Month], 
            'town':[Town], 
            'flat_type':[Flat_type],
            'storey_range':[Storey_range],
            'floor_area_sqm':[Floor_area_sqm],
            'flat_model':[Flat_model],
            'lease_commence_date':[Lease_commence_date],
            'year':[Year]
            })
    
    try: 
        transformed_input = preprocessor.transform(user_input)
        # Make predictions using the best_model
        prediction = guvi.predict(transformed_input)
        st.write('## :orange[Predicted selling price:] ',prediction[0])
    except:
        st.write(':orange[You have entered an invalid value]')
     
st.write(':blue[SKILLS TAKE AWAY FROM THIS PROJECT :]')
st.markdown('Data Wrangling, EDA, Model Building, Model Deployment')
st.markdown(':blue[Data Source :]')
st.link_button("DOWNLOAD THE DATA SOURCE","https://beta.data.gov.sg/collections/189/view")
st.write(':blue[STEPS INVOLVE :]')
st.caption('''Data Collection and Preprocessing: Collect a dataset of resale flat transactions from the Singapore 
           Housing and Development Board (HDB) for the years 1990 to Till Date. Preprocess the data to clean and 
           structure it for machine learning''')

st.caption('''Feature Engineering: Extract relevant features from the dataset, including town, flat type, storey 
           range, floor area, flat model, and lease commence date. Create any additional features that may enhance 
           prediction accuracy. ''')

st.caption('''Model Selection and Training: Choose an appropriate machine learning model for regression 
           (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, 
           using a portion of the dataset for training. ''')

st.caption('''Model Evaluation: Evaluate the model's predictive performance using regression metrics such as Mean 
           Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score. ''')

st.caption('''Streamlit Web Application: Develop a user-friendly web application using Streamlit that allows users
            to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning
            model to predict the resale price based on user inputs. ''')
