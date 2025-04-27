# File: config.py
# Stores database connection settings.

DB_SETTINGS = {
    "database": "holoclean_db",   # Matches POSTGRES_DB in docker-compose.yml
    "user": "holoclean_user",       # Matches POSTGRES_USER in docker-compose.yml
    "password": "holoclean_password", # Matches POSTGRES_PASSWORD in docker-compose.yml
    "host": "localhost",            # Connect to the host machine's mapped port
    "port": "5432"                  # The host port mapped in docker-compose.yml
}

# Add a check or print statement for debugging
print("DB Settings Loaded:")
print(f"  Database: {DB_SETTINGS['database']}")
print(f"  User: {DB_SETTINGS['user']}")
print(f"  Host: {DB_SETTINGS['host']}")
print(f"  Port: {DB_SETTINGS['port']}")
print(f"  Password: {'Set' if DB_SETTINGS['password'] else 'Not Set'}")