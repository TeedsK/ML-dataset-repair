# HoloClean For CS6964

This project by Theo Kremer, Adam Liu, and Luke Aldover was made in order to try and reimplement the HoloClean data cleaning system.

## Setup Instructions

*make sure you have docker installed*

1) Open a terminal in the project's root directory 

2) Run:
```
docker-compose up -d
```
you should see a container named holoclean_postgres_db running and listening on port 5432

3) to build the table definitions, we use the definitions from scehma.sql and run these commands: 

    **Option 1 (we think only works on Mac)**
    ```
    chmod +x init-db.sh
    ./init-db.sh
    ```

    **Option 2**
    ```
    docker cp schema.sql holoclean_postgres_db:/schema.sql
    docker exec -e PGPASSWORD=holoclean_password holoclean_postgres_db psql -U holoclean_user -d holoclean_db -f /schema.sql
    ```

It should output 'Database schema applied successfully.' and no errors should show, you can also check by running:

```
docker exec -it holoclean_postgres_db psql -U holoclean_user -d holoclean_db`
```

and then you can use \dt inside the container

*you can use \q to exit from this view*

4) Opening a virtual env (possibly optional)
```
python -m venv .venv
```
    **On Mac, Run:**
    ```
    source .venv/bin/activate
    ```

    **On Windows, Run:**
    ```
    .venv\Scripts\activate
    ```

then run:
```
pip install -r requirements.txt
```
## How to Run HoloClean 

If you want to run everything at once, run:

```
python ingest.py
python run_detectors.py    
python run_pruning.py     
python run_compiler.py       
python run_inference.py --mode train_predict --learniter 25 --save_model_path trained_model_100.pth --save_builder_path builder_state_100.pkl --pred_output_file marginals_100_rows.pkl --lr 0.005
python evaluate.py --pred_file marginals_100_rows.pkl --truth_file hospital_100_clean.csv
```

and everything will run sequentially and end with the evaluation results


## If You Want to Run One Item at a time

1) insert all the data into the database
```
python ingest.py
```

2) runs the error detectors
```
python run_detectors.py
```

3) runs the pruning and generates domains
```
python run_pruning.py
```

4) runs the compiler and generates the features
```
python run_compiler.py
```

5) runs inference, in training + predicting mode
With these arguments, it creates 25 epochs, has a learning rate of 0.005, and saves the files in their corresponding paths
```
python run_inference.py \
    --mode train_predict \
    --learniter 25 \
    --save_model_path trained_model_100.pth \
    --save_builder_path builder_state_100.pkl \
    --pred_output_file marginals_100_rows.pkl \
    --lr 0.005
```

6) runs the evaluation
With these arguments, it opens the same file generated in step 5 and compares it to the clean hospital data file (our truth file)
```
python evaluate.py \
    --pred_file marginals_100_rows.pkl \
    --truth_file hospital_100_clean.csv
```
