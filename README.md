# HoloClean Re-Implementation Project

This project aims to re-implement the HoloClean data cleaning system.

## Setup Instructions

### Prerequisites

1.  **Docker:** Install Docker Desktop (Windows/Mac) or Docker Engine (Linux). See [Get Docker](https://docs.docker.com/get-docker/).
2.  **Docker Compose:** Usually included with Docker Desktop. For Linux, you might need to install it separately. See [Install Docker Compose](https://docs.docker.com/compose/install/).
3.  **Python:** Python 3.10+ installed.
4.  **Python Dependencies:** Install required libraries:
    ```bash
    pip install pandas "psycopg2-binary>=2.8"
    ```
5.  **Input Data:** Ensure `hospital.csv` is present in the project's root directory (or update the path in `ingest.py`).

### Steps

1.  **Start the Database Container:**
    Open your terminal in the project's root directory (where `docker-compose.yml` is located) and run:
    ```bash
    docker-compose up -d
    ```
    * `-d` runs the container in detached mode (in the background).
    * The first time you run this, Docker will download the `postgres:15` image.
    * This command starts the PostgreSQL container, creates the database (`holoclean_db`), the user (`holoclean_user`), and sets the password (`holoclean_password`).
    * **Check:** Run `docker ps`. You should see a container named `holoclean_postgres_db` running and listing port `5432`.

2.  **Create the Database Schema:**
    You need to apply the table definitions from `schema.sql` to the running database container.
    * **Method A (Recommended - using script):** Make the script executable and run it:
        ```bash
        chmod +x init-db.sh
        ./init-db.sh
        ```
    * **Method B (Manual):** If you don't use the script, run this command:
        ```bash
        docker cp schema.sql holoclean_postgres_db:/schema.sql
        docker exec -e PGPASSWORD=holoclean_password holoclean_postgres_db psql -U holoclean_user -d holoclean_db -f /schema.sql
        # Optional cleanup: docker exec holoclean_postgres_db rm /schema.sql
        ```
    * **Check:** The script/command should output "Database schema applied successfully." or similar. No errors should appear. You can manually verify by running `docker exec -it holoclean_postgres_db psql -U holoclean_user -d holoclean_db` and then using `\dt` inside the container's `psql` prompt (type `\q` to exit `psql`).

3.  **Ingest Initial Data:**
    Run the Python script to load `hospital.csv` into the `cells` table:
    ```bash
    python ingest.py
    ```
    * **Check:**
        * The script should output success messages, including the number of rows inserted.
        * Connect to the database inside the container (as shown in Step 2 Check) and run SQL queries:
            * `SELECT COUNT(*) FROM cells;` (Verify the count matches expected: #rows * #cols)
            * `SELECT COUNT(DISTINCT tid) FROM cells;` (Verify the count matches #rows in CSV)
            * `SELECT * FROM cells LIMIT 5;` (Inspect sample data)

## Command for venv on Mac
```
python -m venv .venv
source .venv/bin/activate

```

## Order to run commands
```
python ingest.py
# add --limit 300 to limit it to the first 300 rows of data

# 2. Run Error Detectors
python run_detectors.py

# 3. Run Pruning (generates domains - uses features)
python run_pruning.py

# 4. Run Compiler (generates features - uses detected errors)
python run_compiler.py

# 5. Run Inference (uses domains and features)
#    (Use fewer samples/burn-in for faster testing if needed)
python run_inference.py --samples 50 --burnin 10
```

### Stopping the Database

To stop the database container when you're done:
```bash
docker-compose down