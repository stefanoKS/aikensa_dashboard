import sqlite3
import mysql.connector


sqlite_conn = sqlite3.connect('./database_results.db')
sqlite_cursor = sqlite_conn.cursor()

# Connect to the MySQL server
mysql_conn = mysql.connector.connect(
    host="10.1.6.253",
    user="AIMACHINE",
    password="Hiroka()07",
    database="AIKENSAresults",
    port = 3306
)
mysql_cursor = mysql_conn.cursor()

# SQL query to check if a record already exists in MySQL
check_query = """
    SELECT id FROM inspection_results
    WHERE partName = %s AND timestampHour = %s AND timestampDate = %s
"""

# SQL query to insert a record into MySQL
insert_query = """
    INSERT INTO inspection_results (
        partName, numofPart, currentnumofPart, timestampHour, timestampDate, 
        deltaTime, kensainName, detected_pitch, delta_pitch, total_length, 
        resultpitch, status, NGreason
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Query to select all rows from SQLite (adjust table name/columns as necessary)
sqlite_cursor.execute("""
    SELECT partName, numofPart, currentnumofPart, timestampHour, timestampDate, 
           deltaTime, kensainName, detected_pitch, delta_pitch, total_length, 
           resultpitch, status, NGreason
    FROM inspection_results
""")
rows = sqlite_cursor.fetchall()

for row in rows:
    # Unpack values (order should match the SQLite query above)
    (partName, numofPart, currentnumofPart, timestampHour, timestampDate,
     deltaTime, kensainName, detected_pitch, delta_pitch, total_length,
     resultpitch, status, NGreason) = row

    # Check if the record already exists in MySQL
    mysql_cursor.execute(check_query, (partName, timestampHour, timestampDate))
    if mysql_cursor.fetchone():
        # print(f"Skipping record: {partName}, {timestampHour}, {timestampDate} (already exists)")
        continue  # Skip insertion for this row

    # Insert the new record into MySQL
    mysql_cursor.execute(insert_query, row)
    mysql_conn.commit()  # Commit after each insertion (or use a batch commit for efficiency)
    # print(f"Inserted record: {partName}, {timestampHour}, {timestampDate}")

# Close the connections
sqlite_conn.close()
mysql_conn.close()
