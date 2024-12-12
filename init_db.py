import sqlite3

# Function to delete all rows in the sos_alerts table
def delete_all_rows():
    # Connect to the SQLite database
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()

    # Execute the DELETE command
    c.execute('DELETE FROM sos_alerts')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("All rows have been deleted from the sos_alerts table.")

# Call the function
if __name__ == '__main__':
    delete_all_rows()
