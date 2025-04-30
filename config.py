DB_SETTINGS = {
    "database": "holoclean_db",
    "user": "holoclean_user", 
    "password": "holoclean_password",
    "host": "localhost",
    "port": "5432"
}

NULL_REPR_PLACEHOLDER="__NULL__"

PRUNING_CORRELATION_THRESHOLD = 0.1
PRUNING_DOMAIN_COOCCURRENCE_THRESHOLD = 0.005
PRUNING_MAX_CANDIDATES_PER_CORR_ATTR = 10
PRUNING_MAX_DOMAIN_SIZE_PER_CELL = 50
PRUNING_MIN_DOMAIN_SIZE = 2

print("DB Settings Loaded:")
print(f"  Database: {DB_SETTINGS['database']}")
print(f"  User: {DB_SETTINGS['user']}")
print(f"  Host: {DB_SETTINGS['host']}")
print(f"  Port: {DB_SETTINGS['port']}")
print(f"  Password: {'Set' if DB_SETTINGS['password'] else 'Not Set'}")