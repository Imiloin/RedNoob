import sqlite3

# connect to the SQLite database
conn = sqlite3.connect('ExploreData.db')
cursor = conn.cursor()

# get the list of tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# assume only one table exists
table_name = tables[0][0]

# get the column information of the table
cursor.execute(f"PRAGMA table_info({table_name});")
columns = cursor.fetchall()
print("Columns:", columns)

# print the first 16 rows of the table
cursor.execute(f"SELECT * FROM {table_name} LIMIT 16;")
rows = cursor.fetchall()
for row in rows:
    print(row)

# close the connection
conn.close()
