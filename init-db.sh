#!/bin/bash
# File: init-db.sh
# Runs the schema.sql script inside the running PostgreSQL Docker container.

CONTAINER_NAME="holoclean_postgres_db"
SQL_FILE="schema.sql"
DB_NAME="holoclean_db"
DB_USER="holoclean_user"

# Check if container is running
if ! docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Error: Container '${CONTAINER_NAME}' is not running."
  echo "Please start it using 'docker-compose up -d'"
  exit 1
fi

echo "Applying schema from ${SQL_FILE} to database ${DB_NAME} in container ${CONTAINER_NAME}..."

# Copy schema file into container
docker cp "${SQL_FILE}" "${CONTAINER_NAME}:/${SQL_FILE}"

# Execute psql command inside container
# Note: We use the PGPASSWORD environment variable for the password
docker exec -e PGPASSWORD=holoclean_password "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -f "/${SQL_FILE}"

# Optional: Remove the copied file after execution
docker exec "${CONTAINER_NAME}" rm "/${SQL_FILE}"

echo "Database schema applied successfully."