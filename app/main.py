import streamlit as st
from huggingduck import HuggDuckDBStreamlitConnection

# Initialise connection
conn = st.connection("titanic", type=HuggDuckDBStreamlitConnection)

# App layout
st.title("Titanic Data Quality Explorer")
st.subheader(f"Available Tables: {', '.join(conn.tables)}")

# Dataset selector
table = st.selectbox("Choose dataset version", conn.tables)

# Data preview
st.dataframe(conn.preview(table))

# Analysis section
st.header("Age Discrepancy Analysis")
discrepancies = conn.query(f"""
    SELECT 
        original.passenger_id,
        original.age AS reported_age,
        {table}.age AS corrected_age,
        ABS(original.age - {table}.age) AS difference
    FROM original
    JOIN {table} ON original.passenger_id = {table}.passenger_id
    WHERE ABS(original.age - {table}.age) > 2
    ORDER BY difference DESC
""")
st.dataframe(discrepancies)

# Schema inspection
if st.checkbox("Show Schema"):
    st.json(conn.get_schema(table))
