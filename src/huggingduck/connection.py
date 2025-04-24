import os
import sys
from typing import Any, Dict, List

import duckdb
import pandas as pd
import streamlit as st
from datasets import get_dataset_infos, load_dataset
from loguru import logger
from streamlit.connections import BaseConnection


class HuggDuckDBConnection:
    """Core connection class for Hugging Face datasets in DuckDB.

    Args:
        repo_id: Hugging Face dataset identifier (format: username/dataset)
        verbose: Enable debug logging if True
    """

    def __init__(self, repo_id: str, verbose: bool = False):
        self.repo_id = repo_id
        self.verbose = verbose
        self.con = duckdb.connect(":memory:")
        self._setup_logging()
        # self._load_datasets()

    def _setup_logging(self) -> None:
        """Configure Loguru logging level."""
        logger.remove()  # Remove default handler
        level = "DEBUG" if self.verbose else "INFO"
        # Directly output to console/terminal; Streamlit will capture it
        logger.add(
            sys.stderr,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )

    def _load_datasets(self) -> None:
        """Load CSV files from HF repo into DuckDB tables with validation."""
        try:
            logger.info(f"Loading dataset: {self.repo_id}")

            try:
                dataset = load_dataset(self.repo_id)
            except Exception as e:
                logger.error(f"Failed to load dataset: {self.repo_id}. Error: {e}")
                raise

            csv_files = []
            try:
                # Check if the dataset has the 'info' attribute and 'download_checksums'
                if hasattr(dataset, "info") and hasattr(
                    dataset.info, "download_checksums"
                ):
                    files = dataset.info.download_checksums.keys()
                    csv_files = [f for f in files if f.endswith(".csv")]
                else:
                    # If not, assume the dataset is a DatasetDict and its keys are the data files
                    csv_files = [f for f in dataset.keys() if f.endswith(".csv")]
            except Exception as e:
                logger.error(f"Failed to retrieve file list from dataset. Error: {e}")
                raise ValueError("Could not determine CSV files in the dataset") from e

            if not csv_files:
                raise ValueError("No CSV files found in repository")

            for file in csv_files:
                table_name = os.path.splitext(file.split("/")[-1])[0]
                logger.debug(f"Processing file: {file} as table '{table_name}'")

                try:
                    df = load_dataset(self.repo_id, data_files=file).to_pandas()
                    self.con.register(table_name, df)
                    logger.success(f"Created table: {table_name} ({len(df)} rows)")
                except Exception as e:
                    logger.error(f"Failed to load CSV file {file}: {str(e)}")
                    raise ValueError(f"Failed to load CSV file {file}: {str(e)}") from e

        except Exception as e:
            logger.error(f"Failed to load {self.repo_id}: {str(e)}")
            raise

    def view_dataset(self) -> Dict[str, Any]:
        """Provides a human-friendly view of the dataset, including metadata."""
        try:
            logger.info(f"Fetching dataset information for: {self.repo_id}")

            # Load dataset information
            try:
                dataset_info = get_dataset_infos(self.repo_id)
                # Convert DatasetInfo to a serializable dictionary
                dataset_info_serializable = {}
                for key, value in dataset_info.items():
                    dataset_info_serializable[key] = str(value)

            except Exception as e:
                logger.error(f"Failed to load dataset info: {e}")
                dataset_info_serializable = {"error": str(e)}

            # Load a sample of the data to get a sense of the structure
            try:
                dataset = load_dataset(
                    self.repo_id, split="train", streaming=True, num_rows=3
                )
                sample_data = list(dataset.take(3))  # Get the first 3 rows
            except Exception as e:
                logger.warning(f"Failed to load dataset sample: {e}")
                sample_data = "Could not load data sample"

            # Prepare the view
            view = {
                "dataset_id": self.repo_id,
                "dataset_info": dataset_info_serializable,
                "sample_data": str(sample_data),
                "tables": self.tables,  # Add list of tables in the DuckDB connection
            }

            return view

        except Exception as e:
            logger.error(f"Failed to view dataset {self.repo_id}: {e}")
            return {"error": str(e)}

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query against loaded datasets.

        Args:
            sql: SQL query string

        Returns:
            Query results as pandas DataFrame

        Raises:
            duckdb.Error: On invalid SQL syntax
        """
        try:
            logger.debug(f"Executing query: {sql}")
            return self.con.execute(sql).fetchdf()
        except duckdb.Error as e:
            logger.error(f"Query failed: {sql}\nError: {str(e)}")
            raise

    def sql(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return the result as a Pandas DataFrame."""
        try:
            logger.debug(f"Executing SQL: {query}")
            return self.con.execute(query).fetchdf()
        except duckdb.Error as e:
            logger.error(f"SQL execution failed: {query}\nError: {str(e)}")
            raise

    def get_schema(self, table_name: str) -> Dict[str, str]:
        """Retrieve column types for a table.

        Args:
            table_name: Name of table to inspect

        Returns:
            Dictionary of {column_name: data_type}

        Raises:
            ValueError: If table doesn't exist
        """
        if table_name not in self.tables:
            raise ValueError(
                f"Table {table_name} not found. Available tables: {self.tables}"
            )

        return (
            self.con.execute(f"DESCRIBE {table_name}")
            .fetchdf()
            .set_index("column_name")["column_type"]
            .to_dict()
        )

    @property
    def tables(self) -> List[str]:
        """List of available tables in the database."""
        return self.con.execute("SHOW TABLES").fetchdf()["name"].tolist()

    def health_check(self) -> Dict[str, Any]:
        """Verify database integrity.

        Returns:
            Dictionary with health status and table counts
        """
        status = {"healthy": False, "tables": {}, "error": None}

        try:
            for table in self.tables:
                count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                status["tables"][table] = count
            status["healthy"] = True
        except Exception as e:
            status["error"] = str(e)

        return status


class HuggDuckDBStreamlitConnection(BaseConnection[HuggDuckDBConnection]):
    """Streamlit-specific connection with caching and secrets integration."""

    def _connect(self, **kwargs) -> HuggDuckDBConnection:
        """Establish connection with options from secrets or kwargs.

        Args:
            **kwargs: Override secrets with explicit parameters

        Returns:
            Configured HuggDuckDBConnection instance
        """
        repo_id = kwargs.get("repo_id", self._secrets.get("repo_id"))
        verbose = kwargs.get("verbose", self._secrets.get("verbose", False))

        if not repo_id:
            raise ValueError("repo_id required (provide via kwargs or secrets.toml)")

        return HuggDuckDBConnection(repo_id=repo_id, verbose=verbose)

    def query(self, sql: str, ttl: int = 3600) -> pd.DataFrame:
        """Cached query execution with automatic retries.

        Args:
            sql: SQL query string
            ttl: Cache duration in seconds

        Returns:
            Query results as pandas DataFrame
        """

        @logger.catch
        @st.cache_data(ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            return self._instance.query(sql)

        return _query(sql)

    def preview(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Quick table preview with validation.

        Args:
            table_name: Name of table to preview
            limit: Number of rows to return

        Returns:
            First N rows of the table
        """
        if table_name not in self._instance.tables:
            raise ValueError(
                f"Invalid table. Available tables: {self._instance.tables}"
            )

        return self.query(f"SELECT * FROM {table_name} LIMIT {limit}")
