import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("adaboost_model.pkl")

# Simplified categories, products, and types
product_categories = ['Electronics', 'Clothing', 'Books']
products = {
    'Electronics': ['Mobile', 'Laptop'],
    'Clothing': ['T-Shirt', 'Jeans'],
    'Books': ['Novel', 'Textbook']
}
types = {
    'Mobile': ['Android', 'iOS'],
    'Laptop': ['Gaming', 'Ultrabook'],
    'T-Shirt': ['Round Neck', 'V-Neck'],
    'Jeans': ['Slim', 'Regular'],
    'Novel': ['Fiction', 'Mystery'],
    'Textbook': ['Math', 'Science']
}

# LabelEncoder mappings used during training
cat_map = {'Electronics': 2, 'Clothing': 1, 'Books': 0}
prod_map = {'Mobile': 9, 'Laptop': 8, 'T-Shirt': 12, 'Jeans': 6, 'Novel': 10, 'Textbook': 14}
type_map = {
    'Android': 2, 'iOS': 29, 'Gaming': 10, 'Ultrabook': 25, 'Round Neck': 21, 'V-Neck': 26,
    'Slim': 23, 'Regular': 19, 'Fiction': 9, 'Mystery': 17, 'Math': 16, 'Science': 22
}
cart_map = {'No': 0, 'Yes': 1}

# Streamlit UI
st.title("🛒 E-Commerce Purchase Prediction")

# Dropdowns
category = st.selectbox("Select Product Category", product_categories)
product = st.selectbox("Select Product", products[category])
prod_type = st.selectbox("Select Type", types[product])

# Inputs
price = st.number_input("Enter Price", min_value=50.0, value=500.0, step=10.0)
num_clicks = st.slider("Number of Clicks", 0, 20, 3)
added_to_cart = st.radio("Added to Cart?", ['Yes', 'No'])

# Prediction
if st.button("Predict Purchase"):
    # Create input in exact model column order
    input_df = pd.DataFrame([[
        price,
        num_clicks,
        cat_map[category],
        prod_map[product],
        type_map[prod_type],
        cart_map[added_to_cart]
    ]], columns=['price', 'num_clicks', 'product_category', 'product', 'type', 'added_to_cart'])

    prediction = model.predict(input_df)[0]
    result = "✅ Likely to Purchase" if prediction == 1 else "❌ Not Likely to Purchase"
    st.success(result)
