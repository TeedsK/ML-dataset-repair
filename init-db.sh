CONTAINER_NAME="holoclean_postgres_db"
SQL_FILE="schema.sql"
DB_NAME="holoclean_db"
DB_USER="holoclean_user"

if ! docker ps --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Error: Container '${CONTAINER_NAME}' is not running."
  echo "Please start it using 'docker-compose up -d'"
  exit 1
fi

echo "Applying schema from ${SQL_FILE} to database ${DB_NAME} in container ${CONTAINER_NAME}..."

docker cp "${SQL_FILE}" "${CONTAINER_NAME}:/${SQL_FILE}"

docker exec -e PGPASSWORD=holoclean_password "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" -f "/${SQL_FILE}"
docker exec "${CONTAINER_NAME}" rm "/${SQL_FILE}"

echo "Database schema finsihed."