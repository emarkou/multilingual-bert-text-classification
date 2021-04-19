import requests
import streamlit as st

classes = {
    "CCAT": "Corporate/Industrial",
    "ECAT": "Economics",
    "GCAT": "Government/Social",
    "MCAT": "Markets"
}

st.title("Text Classification web app")

document = st.text_area("Paste a document")

if st.button("Classify documents"):
    url = "http://localhost:8080/predict"
    payload = {
        "value": document}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, json=payload, headers=headers).json()
    class_assigned = response["assigned_class"]
    class_description = classes[class_assigned]
    st.write(f'The document was classified as {class_assigned}-{class_description}')

