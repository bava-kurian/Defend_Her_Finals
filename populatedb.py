import sqlite3
from datetime import datetime
import random

# Function to generate random SOS details
def get_random_sos_details():
    sos_types = ["Lone Women SOS", "SOS Gesture Detected", "Armed SOS", "Medical Emergency", "Fire Alert"]
    return random.choice(sos_types)

# Function to generate a random latitude and longitude within a specific range
def get_random_location():
    latitude = round(random.uniform(19.0, 21.0), 6)
    longitude = round(random.uniform(71.0, 73.0), 6)
    return f"{latitude}, {longitude}"

# Function to populate the sos_alerts table with multiple data entries
def populate_multiple_entries(num_entries):
    # Connect to the SQLite database
    conn = sqlite3.connect('sos_alerts.db')
    c = conn.cursor()

    # Generate data entries
    alerts = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for _ in range(num_entries):
        details = get_random_sos_details()
        location = get_random_location()
        alerts.append((current_time, details, location))

    # Insert data into the table
    c.executemany('INSERT INTO sos_alerts (timestamp, details, location) VALUES (?, ?, ?)', alerts)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print(f"{num_entries} SOS alert entries have been inserted successfully!")

# Call the function to insert 30 entries
if __name__ == '__main__':
    populate_multiple_entries(30)