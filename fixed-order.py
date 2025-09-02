import streamlit as st
import pandas as pd
import re
import difflib
from sqlalchemy import create_engine, text, inspect
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# --- Streamlit App Title ---
st.title("LLM to SQL with Fuzzy Correction & GROUP BY Fixes")

# --- Database Connection ---
server = "NAVYA"
database = "CompanyDB"
connection_string = (
    f"mssql+pyodbc://@{server}/{database}"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)
engine = create_engine(connection_string)
db = SQLDatabase(engine)

# --- Get Schema Info ---
inspector = inspect(engine)
tables = inspector.get_table_names()

schema = {}
for t in tables:
    cols = [c["name"] for c in inspector.get_columns(t)]
    schema[t] = cols

# --- Show Schema in Sidebar ---
st.sidebar.subheader("📊 Database Schema")
for t, cols in schema.items():
    st.sidebar.write(f"**{t}**")
    st.sidebar.caption(", ".join(cols))

# --- LLM Setup ---
llm = OllamaLLM(model="llama2", temperature=0)
sql_chain = create_sql_query_chain(llm, db)

# --- User Input ---
question = st.text_input(
    "Enter your question:",
    "Show me the 10 orders with the highest sales amounts."
)

# --- Helper Functions ---
def safe_wrap(name: str) -> str:
    """Safely wrap table/column names with brackets if needed"""
    if not name or not name.strip():
        return name
    
    clean_name = name.strip("[]")
    if not clean_name.strip():
        return clean_name
    
    needs_wrapping = (
        " " in clean_name or 
        "-" in clean_name or
        any(c.islower() for c in clean_name) or
        clean_name.upper() in ['ORDER', 'GROUP', 'BY', 'SELECT', 'FROM', 'WHERE', 'SALES', 'DATA']
    )
    
    return f"[{clean_name}]" if needs_wrapping else clean_name

def closest_match(name, choices):
    """Find closest matching name using fuzzy matching with higher threshold"""
    # Increased cutoff from 0.6 to 0.8 for more precise matching
    match = difflib.get_close_matches(name, choices, n=1, cutoff=0.8)
    return match[0] if match else None

def clean_sql_extraction(raw_output: str) -> str:
    """Extract and clean SQL query from LLM output"""
    # First try to find SQL query markers
    sql_patterns = [
        r'```sql\s*(.*?)\s*```',  # SQL code blocks
        r'```\s*(SELECT.*?)\s*```',  # Generic code blocks with SELECT
        r'(SELECT\s+.*?;)',  # Complete SELECT statements ending with semicolon
        r'(SELECT\s+.*?)(?=\n\n|\n[A-Z]|\nAnswer:|\nExplanation:|$)'  # SELECT until paragraph break or explanation
    ]
    
    for pattern in sql_patterns:
        match = re.search(pattern, raw_output, re.IGNORECASE | re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
            # Basic cleanup
            sql_query = re.sub(r'\s+', ' ', sql_query)  # Normalize whitespace
            if sql_query.upper().startswith('SELECT'):
                return sql_query
    
    # Fallback to line-by-line extraction
    lines = raw_output.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\s*SELECT\b', line, re.IGNORECASE):
            in_sql = True
        if in_sql:
            if re.match(r'^(Answer:|Explanation:|The query|Note:)', line, re.IGNORECASE):
                break
            sql_lines.append(line)
            if line.endswith(';'):
                break
    
    result = ' '.join(sql_lines).strip()
    return result if result else raw_output.strip()

def fix_table_names(sql_query: str, schema_info: dict = None) -> str:
    """Fix table names in SQL query by properly wrapping them with brackets"""
    if not schema_info:
        return sql_query
    
    # Much more restrictive patterns that stop at any SQL keyword
    from_pattern = r'\bFROM\s+([`\[\]\'\"]*[A-Za-z][A-Za-z0-9_\s]*?[`\[\]\'\"]*?)(?=\s+(?:WHERE|GROUP|ORDER|HAVING|UNION|INNER|LEFT|RIGHT|JOIN|ON|AS|\)|;|$))'
    join_pattern = r'\b(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|JOIN)\s+([`\[\]\'\"]*[A-Za-z][A-Za-z0-9_\s]*?[`\[\]\'\"]*?)(?=\s+(?:ON|WHERE|GROUP|ORDER|HAVING|UNION|AS|\)|;|$))'
    
    def replace_table_name(match):
        table_ref = match.group(1).strip()
        
        # Skip if already properly wrapped
        if table_ref.startswith('[') and table_ref.endswith(']'):
            return match.group(0)
        
        # Clean any existing quotes/brackets
        clean_table_ref = re.sub(r'^[`\[\]\'\"]+|[`\[\]\'\"]+$', '', table_ref).strip()
        
        # Try exact match first
        exact_match = None
        for table_name in schema_info.keys():
            if clean_table_ref.lower() == table_name.lower():
                exact_match = table_name
                break
        
        if exact_match:
            wrapped_name = safe_wrap(exact_match)
            return match.group(0).replace(table_ref, wrapped_name)
        else:
            # Try fuzzy match with higher threshold
            fuzzy_match = closest_match(clean_table_ref, list(schema_info.keys()))
            if fuzzy_match:
                wrapped_name = safe_wrap(fuzzy_match)
                st.warning(f"🔄 Table name correction: '{clean_table_ref}' → '{fuzzy_match}'")
                return match.group(0).replace(table_ref, wrapped_name)
        
        # If no match found, wrap original name and warn
        wrapped_name = safe_wrap(clean_table_ref)
        st.warning(f"⚠️ Unknown table: '{clean_table_ref}' - please verify it exists")
        return match.group(0).replace(table_ref, wrapped_name)
    
    sql_query = re.sub(from_pattern, replace_table_name, sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(join_pattern, replace_table_name, sql_query, flags=re.IGNORECASE)
    
    return sql_query

def fix_column_names(sql_query: str, schema_info: dict = None) -> str:
    """Fix column names by wrapping them properly if needed"""
    if not schema_info:
        return sql_query
    
    all_columns = []
    for table_name, columns in schema_info.items():
        all_columns.extend(columns)
    
    # More precise pattern to avoid matching SQL keywords and functions
    # This pattern looks for word boundaries and excludes already bracketed items
    column_pattern = r'\b(?<![\[\"])([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)*)(?![\]\"]|\s*\()\b'
    
    def replace_column_if_needed(match):
        col_name = match.group(1).strip()
        
        # Extended list of SQL keywords to avoid wrapping
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'HAVING', 'AS', 'AND', 'OR', 
            'TOP', 'DESC', 'ASC', 'DISTINCT', 'ALL', 'INNER', 'LEFT', 'RIGHT', 'JOIN', 'ON',
            'UNION', 'INTERSECT', 'EXCEPT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'NULL',
            'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS'
        ]
        sql_functions = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'LEN', 'UPPER', 'LOWER']
        
        if col_name.upper() in sql_keywords + sql_functions:
            return match.group(0)
        
        # Check if it's a number
        if col_name.isdigit():
            return match.group(0)
        
        # Look for exact column match that needs wrapping
        for schema_col in all_columns:
            if col_name.lower() == schema_col.lower() and (' ' in schema_col or schema_col.islower()):
                return safe_wrap(schema_col)
        
        return match.group(0)
    
    return re.sub(column_pattern, replace_column_if_needed, sql_query)

def fix_top_clause_parsing(sql_query: str) -> str:
    """Fix issues where TOP clauses get mixed with column names"""
    # Pattern to catch "TOP n COLUMNNAME" being treated as one column
    top_column_pattern = r'\b(TOP\s+\d+)\s+([A-Za-z_][A-Za-z0-9_]*)\b'
    
    def fix_top_column(match):
        top_clause = match.group(1)
        column_name = match.group(2)
        return f"{top_clause} {column_name}"
    
    sql_query = re.sub(top_column_pattern, fix_top_column, sql_query, flags=re.IGNORECASE)
    
    # Remove brackets around TOP clauses that shouldn't be there
    sql_query = re.sub(r'\[TOP\s+\d+\s+([^\]]+)\]', r'TOP \1', sql_query, flags=re.IGNORECASE)
    
    return sql_query

def auto_correct_sql(sql_query: str, schema_info: dict = None) -> str:
    """Cleans and auto-corrects a generated SQL query string for SQL Server"""
    # Basic cleanup
    sql_query = " ".join(sql_query.split())
    sql_query = sql_query.replace(";;", ";")
    
    # Fix obvious structural issues first
    # Fix cases where clauses are merged incorrectly (like "FROMtable" or "GROUP BYcolumn")
    sql_query = re.sub(r'(\w)(FROM|WHERE|GROUP|ORDER|HAVING)\b', r'\1 \2', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\b(FROM|WHERE|GROUP BY|ORDER BY|HAVING)(\w)', r'\1 \2', sql_query, flags=re.IGNORECASE)
    
    # Fix TOP clause issues first
    sql_query = fix_top_clause_parsing(sql_query)
    
    # Pre-validate table names to avoid incorrect corrections
    # If we see obvious malformed table references, try to fix them before table name correction
    malformed_table_pattern = r'FROM\s+\[([^\]]*?(?:GROUP|ORDER|WHERE|HAVING)[^\]]*?)\]'
    if re.search(malformed_table_pattern, sql_query, re.IGNORECASE):
        # Extract just the table name part before any SQL keywords
        def fix_malformed_table(match):
            content = match.group(1)
            # Find the actual table name before any SQL keywords
            table_match = re.search(r'^([^G]*?)(?:\s+(?:GROUP|ORDER|WHERE|HAVING))', content, re.IGNORECASE)
            if table_match:
                clean_table = table_match.group(1).strip()
                return f"FROM [{clean_table}]"
            return match.group(0)
        
        sql_query = re.sub(malformed_table_pattern, fix_malformed_table, sql_query, flags=re.IGNORECASE)
    
    # Fix table and column names
    sql_query = fix_table_names(sql_query, schema_info)
    sql_query = fix_column_names(sql_query, schema_info)
    
    # Fix bracket imbalances
    open_brackets = sql_query.count("[")
    close_brackets = sql_query.count("]")
    if open_brackets > close_brackets:
        sql_query += "]" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        sql_query = "[" * (close_brackets - open_brackets) + sql_query
    
    # Fix TOP clause formatting
    sql_query = re.sub(r"TOP\s*\(\s*(\d+)\s*\)", r"TOP \1", sql_query, flags=re.IGNORECASE)
    
    # Fix common GROUP BY and ORDER BY mistakes
    sql_query = re.sub(r"GROUP BY\s+([^;]+?)\s+BY\s+([^;]+)", r"GROUP BY \1 ORDER BY \2", sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r"GROUP\s+BY\s+ORDER\s+BY", "ORDER BY", sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r"ORDER\s+BY\s+GROUP\s+BY", "GROUP BY", sql_query, flags=re.IGNORECASE)
    
    # Clean up spacing and formatting
    sql_query = re.sub(r",\s*", ", ", sql_query)
    sql_query = re.sub(r"(GROUP BY.*?)(GROUP BY)", r"\1", sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r",\s*(FROM|WHERE|GROUP BY|ORDER BY)", r" \1", sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r"\s*=\s*", " = ", sql_query)
    
    # Ensure proper semicolon ending
    sql_query = sql_query.rstrip(";")
    sql_query = sql_query.strip() + ";"
    
    return sql_query

def fix_group_by(sql_query: str) -> str:
    """Adds missing GROUP BY when aggregate functions are used and fixes GROUP BY column issues"""
    agg_funcs = ["SUM", "COUNT", "AVG", "MIN", "MAX"]
    has_agg = any(re.search(rf"\b{func}\s*\(", sql_query, re.IGNORECASE) for func in agg_funcs)
    
    if not has_agg:
        return sql_query
    
    # Extract SELECT clause
    select_match = re.search(r"SELECT\s+(TOP\s+\d+\s+)?(.+?)\s+FROM", sql_query, re.IGNORECASE | re.DOTALL)
    if not select_match:
        return sql_query
    
    select_clause = select_match.group(2).strip()
    select_items = [item.strip() for item in select_clause.split(",")]
    
    non_agg_cols = []
    for item in select_items:
        # Skip items that contain aggregate functions
        if any(re.search(rf"\b{func}\s*\(", item, re.IGNORECASE) for func in agg_funcs):
            continue
            
        # Extract column name (handle AS aliases)
        col_part = item.split(" AS ")[0].strip() if " AS " in item.upper() else item.strip()
        
        # Skip if it's a literal value or expression
        if col_part.isdigit() or "'" in col_part or '"' in col_part:
            continue
            
        # Extract actual column name, avoiding brackets around aggregate functions
        col_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)*)", col_part)
        if col_match:
            col_name = col_match.group(1).strip()
            
            # Don't add SQL keywords or functions to GROUP BY
            sql_keywords = ['TOP', 'DISTINCT', 'ALL']
            if col_name.upper() not in sql_keywords:
                non_agg_cols.append(safe_wrap(col_name))
    
    # Add GROUP BY if we have non-aggregate columns
    if non_agg_cols:
        # Remove any existing incorrect GROUP BY
        sql_query = re.sub(r"\s+GROUP BY\s+[^;]+?(?=\s+ORDER BY|;|$)", "", sql_query, flags=re.IGNORECASE)
        
        # Add correct GROUP BY clause
        group_by_clause = f" GROUP BY {', '.join(non_agg_cols)}"
        
        if "ORDER BY" in sql_query.upper():
            sql_query = re.sub(r"(\s+ORDER BY)", group_by_clause + r"\1", sql_query, flags=re.IGNORECASE)
        else:
            sql_query = sql_query.rstrip(';') + group_by_clause + ";"
    
    return sql_query

def clean_sql(sql_query: str, schema_info: dict = None) -> str:
    """Runs both auto_correct_sql and fix_group_by"""
    sql_query = auto_correct_sql(sql_query, schema_info)
    sql_query = fix_group_by(sql_query)
    return sql_query

# --- Run Query ---
if st.button("Run Query"):
    st.info("Generating SQL query... ⏳")
    try:
        raw_output = sql_chain.invoke({"question": question})
        st.text_area("Raw LLM Output", raw_output, height=150)

        sql_query = clean_sql_extraction(raw_output)
        if not sql_query or not sql_query.strip().upper().startswith('SELECT'):
            raise ValueError("Could not extract valid SQL query from LLM output")
        
        st.text_area("Extracted SQL Query", sql_query, height=100)

        corrected_sql = clean_sql(sql_query, schema)
        execution_sql = corrected_sql.rstrip(';')
        
        st.text_area("Final Corrected SQL Query", corrected_sql, height=150)

        with engine.connect() as conn:
            result_proxy = conn.execute(text(execution_sql))
            result = result_proxy.fetchall()
            columns = result_proxy.keys()

        df = pd.DataFrame(result, columns=columns)
        st.success("✅ Query executed successfully!")
        st.dataframe(df)
        st.info(f"📊 Retrieved {len(df)} rows")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        if 'sql_query' in locals():
            st.text_area("Debug: Extracted Query", sql_query, height=100)
        if 'corrected_sql' in locals():
            st.text_area("Debug: Corrected Query", corrected_sql, height=100)

# --- Test Section ---
if st.sidebar.checkbox("Show Test Example"):
    st.sidebar.subheader("🧪 Test Example")
    test_query = """
    SELECT TOP 1 ORDERDATE, SALES 
    FROM Autod
    WHERE DAYS_SINCE_LASTORDER = (SELECT MAX(DAYS_SINCE_LASTORDER) FROM Autod)
    """
    
    st.sidebar.text("Original Query:")
    st.sidebar.code(test_query)
    
    fixed_query = clean_sql(test_query, schema)
    st.sidebar.text("Fixed Query:")
    st.sidebar.code(fixed_query)