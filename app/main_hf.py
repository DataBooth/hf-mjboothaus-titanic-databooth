import streamlit as st
import duckdb
import pandas as pd
from streamlit.connections import BaseConnection
from typing import Optional, List
from huggingface_hub import HfApi


class HuggDuckDBConnection(BaseConnection[duckdb.DuckDBPyConnection]):
    """
    Streamlit-specific connection class that leverages DuckDB's native hf:// protocol
    for accessing Hugging Face datasets.
    """

    def _connect(
        self, dataset_path: Optional[str] = None, **kwargs
    ) -> duckdb.DuckDBPyConnection:
        """
        Establishes a DuckDB connection. The dataset_path is expected to be a
        Hugging Face dataset path in the format "hf://datasets/<repo_id>/<filename>".

        Args:
            dataset_path (str, optional): Hugging Face dataset path. Defaults to None.
            **kwargs: Additional arguments (e.g., secrets) are ignored but allowed for flexibility.

        Returns:
            duckdb.DuckDBPyConnection: A DuckDB connection instance.
        """
        try:
            con = duckdb.connect(database=":memory:", read_only=False)

            if dataset_path:
                st.info(f"Connected to Hugging Face dataset: {dataset_path}")
            else:
                st.warning(
                    "No Hugging Face dataset path provided.  Connect will return a DuckDB connection but not point to any specific dataset"
                )
            return con
        except Exception as e:
            st.error(f"Connection error: {e}")
            raise

    def query(self, sql: str, ttl: int = 3600, **kwargs) -> pd.DataFrame:
        """
        Executes a SQL query against the Hugging Face dataset using DuckDB, with caching.

        Args:
            sql (str): The SQL query to execute.
            ttl (int, optional): Cache time-to-live in seconds. Defaults to 3600.
            **kwargs: Additional arguments (e.g., secrets) are ignored but allowed for flexibility.

        Returns:
            pd.DataFrame: The result of the query as a Pandas DataFrame.
        """

        @st.cache_data(ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            try:
                con = self._instance  # Access the DuckDB connection
                df = con.execute(sql).fetchdf()
                st.info(f"Successfully executed query:\n{sql}")
                return df
            except Exception as e:
                st.error(f"Query execution error: {e}")
                raise

        return _query(sql)

    def list_files_in_huggingface_repo(self, repo_id: str) -> List[str]:
        """Lists the files in a Hugging Face dataset repository using the Hugging Face Hub API.

        Args:
            repo_id (str): The Hugging Face repository ID (e.g., "mjboothaus/titanic-databooth").

        Returns:
            List[str]: A list of file names in the repository.
        """
        try:
            api = HfApi()
            repo_files = api.list_repo_files(repo_id, repo_type="dataset")
            return repo_files
        except Exception as e:
            print(f"An error occurred: {e}")
            return []


# Example usage in a Streamlit app:
if __name__ == "__main__":
    # You can define the dataset path in your secrets.toml or pass it directly
    # dataset_path = st.secrets["hf_dataset_path"]
    dataset_path = "hf://datasets/mjboothaus/titanic-databooth/titanic3.csv"

    conn = st.connection(
        "my_duckdb_connection", type=HuggDuckDBConnection, dataset_path=dataset_path
    )

    try:
        df = conn.query(f"SELECT * FROM '{dataset_path}'")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Query error: {e}")

    repo_id = "mjboothaus/titanic-databooth"
    files = conn.list_files_in_huggingface_repo(repo_id)

    if files:
        st.write(files)
    else:
        st.error(f"Could not retrieve file list for repository '{repo_id}'.")
