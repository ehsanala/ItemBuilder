import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Constants
PASSWORD = "MindGames2025!"
API_KEY = st.secrets["BARCODE_LOOKUP_API_KEY"] if "BARCODE_LOOKUP_API_KEY" in st.secrets else ""
API_URL_TEMPLATE = 'https://api.barcodelookup.com/v3/products?barcode={}&formatted=y&key={}"

# Session State
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Password Authentication
if not st.session_state["authenticated"]:
    st.title("🔐 MindGames Item Builder")
    password = st.text_input("Enter Password", type="password")
    if password == PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    else:
        st.stop()

# Main App
st.title("🧩 MindGames Item Builder")
upc_file = st.file_uploader("Upload UPC CSV", type="csv")
category_file = st.file_uploader("Upload Category Mapping CSV", type="csv")
supplier_file = st.file_uploader("Optional: Upload Supplier Backup CSV", type="csv")

# Load AI model if available
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
else:
    model, vectorizer = None, None

if st.button("Generate Items"):
    if upc_file and category_file:
        upcs_df = pd.read_csv(upc_file)
        categories_df = pd.read_csv(category_file)
        supplier_df = pd.read_csv(supplier_file) if supplier_file else pd.DataFrame()

        def map_category(vendor_category):
            for _, row in categories_df.iterrows():
                if pd.notna(row['Sub-Cat 1']) and str(row['Sub-Cat 1']).lower() in vendor_category.lower():
                    return row['Category'], row['Sub-Cat 1'], row['Sub-Cat 2'], row['Sub-Cat 3']
            return 'Uncategorized', '', '', ''

        def get_api_data(upc):
            try:
                response = requests.get(API_URL_TEMPLATE.format(upc, API_KEY))
                if response.status_code == 200:
                    data = response.json()
                    if data.get("products"):
                        return data["products"][0]
            except:
                return None
            return None

        enriched_items = []
        for upc in upcs_df['UPC']:
            product = get_api_data(upc)
            if product:
                title = product.get("title", "")[:60]
                brand = product.get("brand", "")
                vendor_category = product.get("category", "")
                msrp = next((store.get("price", "") for store in product.get("stores", []) if store.get("price")), "")
                images = product.get("images", ["", "", ""])[:3] + ["", ""]
                images = images[:3]
                category, subcat1, subcat2, subcat3 = map_category(vendor_category)
            else:
                fallback = supplier_df[supplier_df['UPC'] == upc]
                title = fallback['Item Name'].values[0] if not fallback.empty else ''
                brand = fallback['Brand'].values[0] if not fallback.empty else ''
                vendor_category = fallback['Category'].values[0] if not fallback.empty else ''
                msrp = fallback['MSRP'].values[0] if not fallback.empty else ''
                images = [fallback['Image 1'].values[0] if not fallback.empty else '', '', '']
                category, subcat1, subcat2, subcat3 = map_category(vendor_category)

            description = title
            if model and vectorizer:
                X = vectorizer.transform([title])
                predicted_category = model.predict(X)[0]
                if predicted_category:
                    category = predicted_category

            extended_desc = f"{title} by {brand}. Part of the {subcat1 or category} collection at MindGames.ca."

            enriched_items.append({
                'UPC': upc,
                'Item Name': title,
                'Description': description,
                'Extended Description': extended_desc,
                'Brand': brand,
                'Category': category,
                'Sub-Category 1': subcat1,
                'Sub-Category 2': subcat2,
                'Sub-Category 3': subcat3,
                'MSRP': msrp,
                'Image 1': images[0],
                'Image 2': images[1],
                'Image 3': images[2],
            })

        output_df = pd.DataFrame(enriched_items)
        st.success("✅ Item generation complete!")
        st.dataframe(output_df)

        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download NetSuite-Ready CSV",
            data=csv,
            file_name='mindgames_items.csv',
            mime='text/csv',
        )
    else:
        st.error("Please upload both the UPC and Category mapping files.")
