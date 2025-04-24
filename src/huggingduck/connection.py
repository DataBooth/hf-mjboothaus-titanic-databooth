import os
from typing import List, Optional, Union

import duckdb
import pandas as pd
import streamlit as st
from huggingface_hub import HfApi
from loguru import logger
from streamlit.connections import BaseConnection


class HuggingDuckDBConnection:
    """
    Core class for interacting with Hugging Face datasets in DuckDB, without Streamlit dependencies.
    """

    def __init__(
        self,
        repo_id: str,
        db_path: Optional[str] = None,
        file_filters: Union[str, List[str]] = None,
        force_recreate: bool = False,
    ):
        """
        Initializes the HuggingDuckDBConnection connection.

        Args:
            repo_id (str): Hugging Face repository ID (e.g., "mjboothaus/titanic-databooth").
            db_path (str, optional): Path to the .duckdb file for persistence. Defaults to None (in-memory).
            file_filters (Union[str, List[str]], optional): File extensions to filter by. Defaults to None.
            force_recreate (bool, optional): Whether to force recreation of the database. Defaults to False.
        """
        self.repo_id = repo_id
        self.db_path = db_path
        self.file_filters = file_filters
        self.force_recreate = force_recreate
        self.con = self._connect()  # Establish connection immediately

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """
        Establishes a DuckDB connection and sets up the schema, with optional persistence.

        Returns:
            duckdb.DuckDBPyConnection: A DuckDB connection instance.
        """
        try:
            # Determine connection method
            if self.db_path:
                db_exists = os.path.exists(self.db_path)
                con = duckdb.connect(database=self.db_path, read_only=False)
                logger.info(f"Connected to DuckDB database file: {self.db_path}")
            else:
                db_exists = False  # In memory databases do not exist
                con = duckdb.connect(database=":memory:", read_only=False)
                logger.info("Connected to in-memory DuckDB database.")

            self.schema_name = self.repo_id.replace("/", "_").replace(
                "-", "_"
            )  # Create a valid schema name

            # Check if we should recreate the database
            if self.force_recreate or not db_exists:
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
                self._load_all_datasets(con)
            else:
                # Database exists, skip loading data
                con.execute(
                    f"SET search_path = '{self.schema_name}';"
                )  # Set schema as the default
                logger.info(
                    f"Using existing database.  Skipping data load from Hugging Face. To force reload, set force_recreate=True"
                )

            return con
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def _load_all_datasets(self, con: duckdb.DuckDBPyConnection):
        """Loads metadata and (optionally) data for all qualifying files in the Hugging Face repo."""
        files = self.list_files_in_huggingface_repo(
            self.repo_id, self.file_filters
        )  # Get filtered list of files

        for file in files:
            try:
                file_path = f"hf://datasets/{self.repo_id}/{file}"
                table_name = os.path.splitext(os.path.basename(file))[
                    0
                ]  # Derive table name from file name

                # Create table for the data, and load the data set
                con.execute(
                    f"CREATE TABLE IF NOT EXISTS {self.schema_name}.{table_name} AS SELECT * FROM '{file_path}'"
                )

                # Get row count using SQL query
                row_count = con.execute(
                    f"SELECT count(*) FROM {self.schema_name}.{table_name}"
                ).fetchone()[0]

                # Get file size  The is is tricky and can be solved as an enhancement
                file_size = 0

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
                logger.error(f"Error loading dataset {file}: {e}")
                con.execute(f"""
                    INSERT INTO {self.schema_name}.metadata (file_name, file_size, file_type, row_count, is_loaded)
                    VALUES ('{file}', 0, 'unknown', 0, FALSE);
                """)

    def query(self, sql: str) -> pd.DataFrame:
        """
        Executes a SQL query against the DuckDB connection.
        """
        try:
            logger.debug(f"Executing query: {sql}")
            df = self.con.execute(sql).fetchdf()
            logger.info(f"Successfully executed query:\n{sql}")
            return df
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    def sql_df(self, sql: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.
        """
        try:
            logger.debug(f"Executing query: {sql}")
            df = self.con.sql(sql).df()
            logger.info(f"Successfully executed query:\n{sql}")
            return df
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

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
            logger.error(f"An error occurred: {e}")
            return []

    def get_table_names(self, exclude_metadata: bool = True) -> List[str]:
        """Retrieves the table names from the DuckDB connection, optionally excluding the metadata table.

        Args:
            exclude_metadata (bool, optional): Whether to exclude the metadata table from the results. Defaults to True.

        Returns:
            List[str]: A list of table names in the database.
        """
        try:
            table_names = self.con.execute("SHOW TABLES").fetchdf()["name"].tolist()
            if exclude_metadata:
                table_names = [name for name in table_names if name != "metadata"]
            return table_names
        except Exception as e:
            logger.error(f"Error retrieving table names: {e}")
            return []

    def close(self):
        """Closes the DuckDB connection."""
        if self.con:
            self.con.close()
            logger.info("DuckDB connection closed.")


class HuggingDuckDBStConnection(BaseConnection[HuggingDuckDBConnection]):
    """
    Streamlit-specific connection class that leverages HuggingDuckDBConnection for interacting with Hugging Face datasets.
    """

    def _connect(self, **kwargs) -> HuggingDuckDBConnection:
        """
        Establishes the connection using parameters from secrets or kwargs.

        Args:
            **kwargs: Parameters for HuggingDuckDBConnection (repo_id, db_path, file_filters, force_recreate).
        """
        repo_id = kwargs.get("repo_id", st.secrets.get("repo_id"))
        db_path = kwargs.get("db_path", st.secrets.get("db_path"))
        file_filters = kwargs.get("file_filters", st.secrets.get("file_filters"))
        force_recreate = kwargs.get(
            "force_recreate", st.secrets.get("force_recreate", False)
        )

        if not repo_id:
            st.error("repo_id is required (provide via kwargs or secrets.toml)")
            raise ValueError("repo_id is required")

        try:
            hdb = HuggingDuckDBConnection(
                repo_id=repo_id,
                db_path=db_path,
                file_filters=file_filters,
                force_recreate=force_recreate,
            )
            st.info(f"Connected to HuggingFace repo: {repo_id}")
            return hdb
        except Exception as e:
            st.error(f"Connection failed: {e}")
            raise

    def query(self, sql: str, ttl: int = 3600) -> pd.DataFrame:
        """
        Executes a SQL query against the HuggingDuckDBConnection connection, with caching.
        """

        @st.cache_data(ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            try:
                df = self._instance.query(sql)
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
            files = self._instance.list_files_in_huggingface_repo(repo_id, file_filters)
            return files
        except Exception as e:
            st.error(f"Error retrieving table names: {e}")
            return []

    def get_table_names(self, exclude_metadata: bool = True) -> List[str]:
        """Retrieves the table names from the DuckDB connection, optionally excluding the metadata table.

        Args:
            exclude_metadata (bool, optional): Whether to exclude the metadata table from the results. Defaults to True.

        Returns:
            List[str]: A list of table names in the database.
        """
        try:
            table_names = self._instance.get_table_names(exclude_metadata)
            return table_names
        except Exception as e:
            st.error(f"Error retrieving table names: {e}")
            return []

    def close(self):
        """Closes the DuckDB connection."""
        self._instance.close()  # Call the close method of the HuggingDuckDBConnection instance
