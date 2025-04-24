import streamlit as st
import duckdb
import pandas as pd
from streamlit.connections import BaseConnection
from typing import Optional, List, Union
from huggingface_hub import HfApi
import os
import io


class HuggDuckDBConnection(BaseConnection[duckdb.DuckDBPyConnection]):
    """
    Streamlit-specific connection class that leverages DuckDB with schema/namespace support
    and persistence to a .duckdb file.
    """

    def _connect(
        self,
        repo_id: str,
        db_path: Optional[str] = None,
        file_filters: Union[str, List[str]] = None,
        **kwargs,
    ) -> duckdb.DuckDBPyConnection:
        """
        Establishes a DuckDB connection and sets up the schema, with optional persistence.

        Args:
            repo_id (str): Hugging Face repository ID (e.g., "mjboothaus/titanic-databooth").
            db_path (str, optional): Path to the .duckdb file for persistence. Defaults to None (in-memory).
            file_filters (Union[str, List[str]], optional): File extensions to filter by. Defaults to None.
            **kwargs: Additional arguments (e.g., secrets) are ignored but allowed for flexibility.

        Returns:
            duckdb.DuckDBPyConnection: A DuckDB connection instance.
        """
        try:
            # Connect to DuckDB (either in-memory or persistent)
            if db_path:
                con = duckdb.connect(database=db_path, read_only=False)
                st.info(f"Connected to DuckDB database file: {db_path}")
            else:
                con = duckdb.connect(database=":memory:", read_only=False)
                st.info("Connected to in-memory DuckDB database.")

            self.repo_id = repo_id  # Store repo_id as an instance variable
            self.schema_name = repo_id.replace("/", "_").replace(
                "-", "_"
            )  # Create a valid schema name
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
                    is_loaded BOOLEAN
                );
            """)

            # Load metadata and (optionally) data for all files in the repo
            self._load_all_datasets(con, repo_id, file_filters)

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
                table_name = os.path.splitext(os.path.basename(file))[
                    0
                ]  # Derive table name from file name

                # Load data into a Pandas DataFrame
                df = con.execute(f"SELECT * FROM '{file_path}'").fetchdf()

                # Serialize to CSV using StringIO
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()

                # Get file size
                file_size = len(csv_string.encode("utf-8"))  # Get size in bytes

                # Get file type
                file_type = file.split(".")[-1].lower()

                # Create table for the data, and load the data set
                con.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} AS SELECT * FROM '{file_path}'"
                )

                # Add metadata
                con.execute(f"""
                    INSERT INTO {self.schema_name}.metadata (file_name, file_size, file_type, is_loaded)
                    VALUES ('{file}', {file_size}, '{file_type}', TRUE);
                """)
                st.success(
                    f"Loaded dataset into table: {self.schema_name}.{table_name}"
                )

            except Exception as e:
                st.error(f"Error loading dataset {file}: {e}")
                con.execute(f"""
                    INSERT INTO {self.schema_name}.metadata (file_name, file_size, file_type, is_loaded)
                    VALUES ('{file}', 0, 'unknown', FALSE);
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
                st.info(f"Successfully executed query:\n{sql}")
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


# Example usage in a Streamlit app:
if __name__ == "__main__":
    repo_id = "mjboothaus/titanic-databooth"  # Define the repo_id
    db_path = "titanic.duckdb"  # Specify the path to the .duckdb file

    conn = st.connection(
        "my_duckdb_connection",
        type=HuggDuckDBConnection,
        repo_id=repo_id,
        db_path=db_path,
        file_filters=["csv"],
    )

    # Example query (replace with your actual query)
    try:
        df = conn.query(f"SELECT * FROM {conn.schema_name}.train LIMIT 10")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Query error: {e}")

    # Display metadata
    try:
        metadata = conn.query(f"SELECT * FROM {conn.schema_name}.metadata")
        st.write("Metadata:", metadata)
    except Exception as e:
        st.error(f"Error fetching metadata: {e}")
