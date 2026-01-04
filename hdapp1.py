import streamlit as st
import pickle 
import numpy as np 

st.title('Welcome to Heart Disease Prediction!')

# Load the pretrained model
with open('heart_disease.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

st.image('heart.jpeg', width=500, caption='Heart Disease Prediction App Image')

st.sidebar.header('How to Use:')
st.sidebar.markdown(
    """
 1. Enter the Patient's Details 
 2. Click "Predict" to find out the risk
 3. The app uses a trained ML model to help you make informed health decisions!
 
**DISCLAIMER!**  
Consult a doctor for more info!
    """
)

st.header('Enter Patient Details')

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=45)
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3, 4], 
                      help='0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic, 4: Other')

with col2:
    trestbps = st.number_input('Resting Blood Pressure (trestbps) mm Hg', 
                               min_value=80, max_value=200, value=120)
    chol = st.number_input('Serum Cholesterol (chol) mg/dl', 
                          min_value=100, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1],
                       format_func=lambda x: 'No' if x == 0 else 'Yes')

with col3:
    restecg = st.selectbox('Resting ECG Results (restecg)', [0, 1, 2],
                          help='0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy')
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', 
                              min_value=60, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1],
                        format_func=lambda x: 'No' if x == 0 else 'Yes')

# Add some spacing
st.markdown("---")

if st.button('Predict', type='primary'):
    # Preparing the input for the model with all 9 features
    # Order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display results with better formatting
    st.markdown("### Prediction Results")
    
    # Define risk levels (0 = no disease, 1-4 = increasing severity)
    risk_levels = {
        0: {
            'title': '✅ No Heart Disease Detected',
            'message': 'The patient shows **no signs** of heart disease.',
            'color': 'success',
            'recommendation': 'Continue maintaining a healthy lifestyle with regular check-ups.',
            'severity': 'None'
        },
        1: {
            'title': '⚠️ Mild Risk Detected',
            'message': 'The patient shows **mild signs** of heart disease.',
            'color': 'info',
            'recommendation': 'Consult with a healthcare professional for preventive measures and lifestyle modifications.',
            'severity': 'Low'
        },
        2: {
            'title': '⚠️ Moderate Risk Detected',
            'message': 'The patient shows **moderate signs** of heart disease.',
            'color': 'warning',
            'recommendation': 'Medical consultation is recommended. Consider further diagnostic tests and treatment planning.',
            'severity': 'Moderate'
        },
        3: {
            'title': '🚨 High Risk Detected',
            'message': 'The patient shows **significant signs** of heart disease.',
            'color': 'warning',
            'recommendation': 'Seek medical attention soon. Comprehensive cardiovascular evaluation is strongly advised.',
            'severity': 'High'
        },
        4: {
            'title': '🚨 Very High Risk Detected',
            'message': 'The patient shows **severe signs** of heart disease.',
            'color': 'error',
            'recommendation': 'URGENT: Immediate medical consultation required. Do not delay seeking professional medical care.',
            'severity': 'Very High'
        }
    }
    
    # Get the risk level information
    risk_info = risk_levels.get(prediction, risk_levels[0])
    
    # Display based on severity
    st.markdown(f"#### {risk_info['title']}")
    
    if prediction == 0:
        st.success(risk_info['message'])
        st.balloons()
    elif prediction in [1, 2]:
        st.warning(risk_info['message'])
    else:  # prediction in [3, 4]
        st.error(risk_info['message'])
    
    # Display severity and recommendation in columns
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Risk Severity Level", value=risk_info['severity'])
    with col_b:
        st.metric(label="Risk Score", value=f"{prediction}/4")
    
    st.info(f"**Recommendation:** {risk_info['recommendation']}")
    
    # Visual risk indicator
    st.markdown("**Risk Level Indicator:**")
    risk_colors = ['🟢', '🟡', '🟠', '🔴', '🔴']
    st.markdown(' '.join(risk_colors[:prediction+1]) + ' ' + '⚪' * (4-prediction))

# Add information section at the bottom
st.markdown("---")
with st.expander("ℹ️ Feature Descriptions"):
    st.markdown("""
    - **Age**: Patient's age in years
    - **Sex**: 0 = Female, 1 = Male
    - **Chest Pain Type (cp)**: Type of chest pain experienced
    - **Resting Blood Pressure (trestbps)**: Blood pressure at rest (mm Hg)
    - **Serum Cholesterol (chol)**: Cholesterol level (mg/dl)
    - **Fasting Blood Sugar (fbs)**: Whether fasting blood sugar > 120 mg/dl
    - **Resting ECG (restecg)**: Resting electrocardiographic results
    - **Max Heart Rate (thalach)**: Maximum heart rate achieved during exercise
    - **Exercise Induced Angina (exang)**: Whether exercise causes chest pain
    """)