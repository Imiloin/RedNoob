import sqlite3

# connect to the SQLite database
conn = sqlite3.connect("ExploreData.db")
cursor = conn.cursor()

# get the list of tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# assume only one table exists
table_name = tables[0][0]

# delete rows where "作者ID" equals a specific value
author_id_to_delete = "68257a85000000000601ff60"  # replace with the author ID to delete
cursor.execute(f"DELETE FROM {table_name} WHERE 作者ID = ?;", (author_id_to_delete,))
conn.commit()  # commit the changes to the database
print(f"Rows with 作者ID = {author_id_to_delete} have been deleted.")

# close the connection
conn.close()
