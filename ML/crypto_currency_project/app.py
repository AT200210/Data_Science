import streamlit as st
import pandas as pd
import numpy as np
import joblib 


st.title('💰 Bitcoin Price Predictor')
st.write('Enter values to predict the Closing Price of Bitcoin')

@st.cache_resource
def load_models():
    try:
        lr_model=joblib.load('linear_regression_model.pkl')
        svr_model=joblib.load('svr_model.pkl')
        tree_model=joblib.load('tree_regression_model.pkl')
        scaler=joblib.load('scaler.pkl')
        return lr_model, svr_model, tree_model, scaler
    except:
        st.error('⚠️ Model files not in memory! Execute the model training code!!')
        return None, None, None, None
    

lr_model, svr_model, tree_model, scaler= load_models()

if lr_model is not None:

# Input section
    st.header("📉 Enter Bitcoin Data")

    col1, col2= st.columns(2)

    with col1:
        high=st.number_input('High Price ($)', value=3000.00, min_value=0.0)
        low = st.number_input("Low Price ($)", value=2800.00, min_value=0.0)
        open_price = st.number_input("Open Price ($)", value=2900.00, min_value=0.0)
    
    with col2:
        volume = st.number_input("Volume", value=1000000.0, min_value=0.0)
        marketcap = st.number_input("Market Cap", value=350000000000.0, min_value=0.0)
    
    model_choice = st.selectbox("Choose Model:", ["Linear Regression", "SVR", 'Decision Tree'])

    if st.button('🔮 Predict Price'):

        input_data=np.array([[high, low, open_price, volume, marketcap]])
        input_scaled=scaler.transform(input_data)

        if model_choice == 'Linear Regression':
            prediction = lr_model.predict(input_scaled)[0]
            st.success(f"🔵 Linear Regression Prediction: **${prediction:,.2f}**")

        elif model_choice == 'SVR':
            prediction = svr_model.predict(input_scaled)[0]
            st.success(f"🔵 SVR Prediction: **${prediction:,.2f}**")

        else:
            prediction = tree_model.predict(input_scaled)[0]
            st.success(f"🔵 Tree Prediction: **${prediction:,.2f}**")

        st.subheader("Your Inputs:")
        input_df = pd.DataFrame({
            'Feature': ['High', 'Low', 'Open', 'Volume', 'Marketcap'],
            'Value': [high, low, open_price, volume, marketcap]
        })
        st.table(input_df)  


st.sidebar.header("ℹ️ Instructions")
st.sidebar.write("""
1. Enter Ethereum market data
2. Choose a prediction model
3. Click 'Predict Price'
4. View your prediction!
""")

st.sidebar.header("📝 About")
st.sidebar.write("""
**Linear Regression**: Simple, fast
**Decison Tree**: More complex, often more accurate
**SVR**: Modular Predictions withsome misclassifications
""")


    