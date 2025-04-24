import streamlit as st
from huggingduck import HuggDuckDBStreamlitConnection, HuggDuckDBConnection
import json

HF_REPO = "mjboothaus/titanic-databooth"

st.set_page_config(layout="wide")


conn_raw = HuggDuckDBConnection(HF_REPO)

dataset_view = conn_raw.view_dataset()

st.json(json.dumps(dataset_view, indent=4))


# # Initialise connection
# conn = st.connection(
#     "titanic",
#     type=HuggDuckDBStreamlitConnection,
#     repo_id=HF_REPO,
# )

# table_names = conn._instance.tables

# st.title("Titanic Data Quality Explorer")
# st.subheader(f"Huggingface.co `{HF_REPO}`")

# tab1, tab2 = st.tabs(["Data", "Schema"])

# with tab1:
#     st.markdown(f"Available Tables: {table_names}")

#     table_select = st.selectbox("Choose dataset version", table_names)

#     if table_select:
#         st.dataframe(conn.preview(table_select))

#     # Analysis section
#     if "original" in table_names:
#         st.header("Age Discrepancy Analysis")
#         discrepancies = conn.query(f"""
#             SELECT
#                 original.passenger_id,
#                 original.age AS reported_age,
#                 {table_select}.age AS corrected_age,
#                 ABS(original.age - {table_select}.age) AS difference
#             FROM original
#             JOIN {table_select} ON original.passenger_id = {table_select}.passenger_id
#             WHERE ABS(original.age - {table_select}.age) > 2
#             ORDER BY difference DESC
#         """)
#         st.dataframe(discrepancies)

# with tab2:
#     st.header("Table schema")
#     # Schema inspection
#     if st.checkbox("Show Schema"):
#         st.json(conn._instance.get_schema(table_select))
