# File: config.py
# Stores database connection settings.

DB_SETTINGS = {
    "database": "holoclean_db",   # Matches POSTGRES_DB in docker-compose.yml
    "user": "holoclean_user",       # Matches POSTGRES_USER in docker-compose.yml
    "password": "holoclean_password", # Matches POSTGRES_PASSWORD in docker-compose.yml
    "host": "localhost",            # Connect to the host machine's mapped port
    "port": "5432"                  # The host port mapped in docker-compose.yml
}

NULL_REPR_PLACEHOLDER="__NULL__"

PRUNING_CORRELATION_THRESHOLD = 0.1 # Minimum correlation score to consider an attribute related
PRUNING_DOMAIN_COOCCURRENCE_THRESHOLD = 0.005 # Minimum P(target_val | cond_val) to consider a candidate
PRUNING_MAX_CANDIDATES_PER_CORR_ATTR = 10 # Max candidates to add from a single correlated attribute
PRUNING_MAX_DOMAIN_SIZE_PER_CELL = 50 # Hard cap on domain size (optional, TensorBuilder handles variable)
PRUNING_MIN_DOMAIN_SIZE = 2 # Try to ensure at least this many candidates (adds frequent if needed)

# Add a check or print statement for debugging
print("DB Settings Loaded:")
print(f"  Database: {DB_SETTINGS['database']}")
print(f"  User: {DB_SETTINGS['user']}")
print(f"  Host: {DB_SETTINGS['host']}")
print(f"  Port: {DB_SETTINGS['port']}")
print(f"  Password: {'Set' if DB_SETTINGS['password'] else 'Not Set'}")