

import marimo

__generated_with = "0.13.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Huggingface to DuckDB Custom Class

        ## Example: Titanic Data Analysis (Marimo notebook)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Overview

        Custom `HuggDuckDBConnection` class to pull data from [Huggingface.co](https://huggingface.co) into a [DuckDB](https://duckdb.org) in-memory database.

        Optional Streamlit-specific `st.connection` class HuggDuckDBStreamlitConnection which inherits from this.

        See example `app/main.py` for details.

        This example uses thee classic Titanic dataset from the `mjboothaus/titanic-databooth` repo.

        <https://www.databooth.com.au>
        """
    )
    return


@app.cell
def _():
    from huggingduck import HuggDuckDBConnection
    return (HuggDuckDBConnection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Initialise connection

        The `conn` object contains a list of the (csv) files discovered in the repo.

        `conn = HuggDuckDBConnection("mjboothaus/titanic-databooth")`
        """
    )
    return


@app.cell
def _(HuggDuckDBConnection, mo):
    conn = HuggDuckDBConnection("mjboothaus/titanic-databooth")

    if conn:
        mo.md(f"## Available Tables: {', '.join(conn.tables)}")
    return (conn,)


@app.cell
def _(conn, mo):
    table_select = mo.ui.dropdown(options=conn.tables, value=conn.tables[0], label="Choose dataset version")

    table_select
    return


@app.cell
def _(conn, mo):
    table = conn.tables[0]
    df = conn.sql(f"SELECT * FROM {table}")
    mo.ui.table(df)
    return (table,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Table schema

        This is the schema for the selected table.
        """
    )
    return


@app.cell
def _(conn, mo, table):
    schema = conn.get_schema(table)

    mo.md(f"### {table} Schema")
    return (schema,)


@app.cell
def _(schema):
    schema
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
