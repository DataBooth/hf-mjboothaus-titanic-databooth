import io
import os
from typing import List, Optional, Union

import duckdb
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from loguru import logger
from streamlit.connections import BaseConnection


class HuggingDuckDBConnection(BaseConnection[duckdb.DuckDBPyConnection]):
    """
    Streamlit-specific connection class that leverages DuckDB with schema/namespace support
    and persistence to a .duckdb file.
    """

    def _connect(
        self,
        repo_id: str,
        db_path: Optional[str] = None,
        file_filters: Union[str, List[str]] = None,
        force_recreate: bool = False,
        **kwargs,
    ) -> duckdb.DuckDBPyConnection:
        """
        Establishes a DuckDB connection and sets up the schema, with optional persistence.

        Args:
            repo_id (str): Hugging Face repository ID (e.g., "mjboothaus/titanic-databooth").
            db_path (str, optional): Path to the .duckdb file for persistence. Defaults to None (in-memory).
            file_filters (Union[str, List[str]], optional): File extensions to filter by. Defaults to None.
            force_recreate (bool, optional): Whether to force recreation of the database. Defaults to False.
            **kwargs: Additional arguments (e.g., secrets) are ignored but allowed for flexibility.

        Returns:
            duckdb.DuckDBPyConnection: A DuckDB connection instance.
        """
        try:
            # Determine connection method
            if db_path:
                db_exists = os.path.exists(db_path)
                con = duckdb.connect(database=db_path, read_only=False)
                logger.info(f"Connected to DuckDB database file: {db_path}")
            else:
                db_exists = False  # In memory databases do not exist
                con = duckdb.connect(database=":memory:", read_only=False)
                logger.info("Connected to in-memory DuckDB database.")

            self.repo_id = repo_id  # Store repo_id as an instance variable
            self.schema_name = repo_id.replace("/", "_").replace(
                "-", "_"
            )  # Create a valid schema name

            # Check if we should recreate the database
            if force_recreate or not db_exists:
                con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name};")
                con.execute(
                    f"SET search_path = '{self.schema_name}';"
                )  # Set schema as the default

                # Create metadata table
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema_name}.metadata (
                        file_name VARCHAR,
                        file_size INTEGER,
                        file_type VARCHAR,
                        row_count INTEGER,
                        is_loaded BOOLEAN
                    );
                """)
                # Load metadata and (optionally) data for all files in the repo
                self._load_all_datasets(con, repo_id, file_filters)
            else:
                # Database exists, skip loading data
                con.execute(
                    f"SET search_path = '{self.schema_name}';"
                )  # Set schema as the default
                logger.info(
                    f"Using existing database.  Skipping data load from Hugging Face.  To force reload, set force_recreate=True"
                )

            return con
        except Exception as e:
            st.error(f"Connection error: {e}")
            raise

    def _load_all_datasets(
        self,
        con: duckdb.DuckDBPyConnection,
        repo_id: str,
        file_filters: Union[str, List[str]] = None,
    ):
        """Loads metadata and (optionally) data for all qualifying files in the Hugging Face repo."""
        files = self.list_files_in_huggingface_repo(
            repo_id, file_filters
        )  # Get filtered list of files

        for file in files:
            try:
                file_path = f"hf://datasets/{repo_id}/{file}"
                table_name = os.path.splitext(os.path.basename(file))[0]

                # Load data into a Pandas DataFrame
                df = con.execute(f"SELECT * FROM '{file_path}'").fetchdf()

                # Serialize to CSV using StringIO
                csv_buffer = io.StringIO()  # Commented out: Not needed for row count
                df.to_csv(
                    csv_buffer, index=False
                )  # Commented out: Not needed for row count
                csv_string = (
                    csv_buffer.getvalue()
                )  # Commented out: Not needed for row count

                # Get file size
                file_size = len(
                    csv_string.encode("utf-8")
                )  # Get size in bytes # Commented out: Not needed for row count

                # Create table for the data, and load the data set
                con.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} AS SELECT * FROM '{file_path}'"
                )

                # Get row count using SQL query
                row_count = con.execute(
                    f"SELECT count(*) FROM {self.schema_name}.{table_name}"
                ).fetchone()[0]

                # Get file type
                file_type = file.split(".")[-1].lower()

                # Add metadata
                con.execute(f"""
                    INSERT INTO {self.schema_name}.metadata (file_name, file_size, file_type, row_count, is_loaded)
                    VALUES ('{file}', {file_size}, '{file_type}', {row_count}, TRUE);
                """)
                logger.info(
                    f"Loaded dataset into table: {self.schema_name}.{table_name}"
                )

            except Exception as e:
                st.error(f"Error loading dataset {file}: {e}")
                con.execute(f"""
                    INSERT INTO {self.schema_name}.metadata (file_name, file_size, file_type, row_count, is_loaded)
                    VALUES ('{file}', 0, 'unknown', 0, FALSE);
                """)

    def query(self, sql: str, ttl: int = 3600, **kwargs) -> pd.DataFrame:
        """
        Executes a SQL query against the DuckDB connection with caching.
        """

        @st.cache_data(ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            try:
                con = self._instance  # Access the DuckDB connection
                df = con.execute(sql).fetchdf()
                logger.info(f"Successfully executed query:\n{sql}")
                return df
            except Exception as e:
                st.error(f"Query execution error: {e}")
                raise

        return _query(sql)

    def list_files_in_huggingface_repo(
        self, repo_id: str, file_filters: Union[str, List[str]] = None
    ) -> List[str]:
        """Lists the files in a Hugging Face dataset repository using the Hugging Face Hub API,
        applying optional file filters.

        Args:
            repo_id (str): The Hugging Face repository ID (e.g., "mjboothaus/titanic-databooth").
            file_filters (str, List[str], optional): A string or a list of strings representing file extensions to filter by
                (e.g., "csv" or ["csv", "parquet"]). Defaults to None (no filter).

        Returns:
            List[str]: A list of file names in the repository that match the specified filters.
        """
        try:
            api = HfApi()
            repo_files = api.list_repo_files(repo_id, repo_type="dataset")

            if file_filters:
                if isinstance(file_filters, str):
                    file_filters = [
                        file_filters
                    ]  # Convert to a list if it's a single string

                filtered_files = []
                for file in repo_files:
                    for file_filter in file_filters:
                        if file.endswith(f".{file_filter}"):
                            filtered_files.append(file)
                            break  # Once a match is found, move to the next file
                return filtered_files
            else:
                return repo_files  # No filter, return all files

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def get_table_names(self, exclude_metadata: bool = True) -> List[str]:
        """Retrieves the table names from the DuckDB connection, optionally excluding the metadata table.

        Args:
            exclude_metadata (bool, optional): Whether to exclude the metadata table from the results. Defaults to True.

        Returns:
            List[str]: A list of table names in the database.
        """
        try:
            con = self._instance  # Access the DuckDB connection
            # Get the table names from the current schema
            table_names = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
            if exclude_metadata:
                table_names = [name for name in table_names if name != "metadata"]
            return table_names
        except Exception as e:
            st.error(f"Error retrieving table names: {e}")
            return []


# Example usage in a Streamlit app:
if __name__ == "__main__":
    HF_REPO = "mjboothaus/titanic-databooth"  # Define the repo_id
    db_path = "titanic.duckdb"  # Specify the path to the .duckdb file

    st.set_page_config(layout="wide")

    st.title("Huggingface.co repo viewer")
    st.subheader(f"`{HF_REPO}`")

    conn = st.connection(
        HF_REPO,
        type=HuggingDuckDBConnection,
        repo_id=HF_REPO,
        db_path=db_path,
        file_filters=["csv"],
        force_recreate=False,
    )

    tab_data, tab_meta = st.tabs(["Data", "Metadata"])

    with tab_data:
        data_table = st.selectbox(label="Select table", options=conn.get_table_names())
        try:
            df = conn.query(f"SELECT * FROM {conn.schema_name}.{data_table}")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Query error: {e}")

    with tab_meta:
        try:
            metadata = conn.query(f"SELECT * FROM {conn.schema_name}.metadata")
            st.write("Metadata:", metadata)
        except Exception as e:
            st.error(f"Error fetching metadata: {e}")
