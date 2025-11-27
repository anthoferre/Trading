from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta
from docker.types import Mount

# --- CONFIGURATION GLOBALE ---
DOCKER_IMAGE_NAME = "ghcr.io/anthoferre/trading:latest"
PROJECT_DIR_HOST = "../.." 

with DAG(
    dag_id="mlops_trading_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 0 * * *",  # Exécution quotidienne a minuit (WFV Entrainement)
    catchup=False,
    default_args={
        "owner": "airflow",
        "email": ["anthony.ferre@hotmail.fr"],
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "trading"],
) as dag:

    train_model_task = DockerOperator(
        task_id="train_walk_forward_validation",
        image=DOCKER_IMAGE_NAME,
        command="python run_pipeline.py train", 
        mounts=[
        Mount(
            source='/opt/airflow/models',   
            target='/app/models',           
            type='bind',                  
            read_only=False                
        ),
        Mount(
            source='/opt/airflow/mlruns',
            target='/app/mlruns',
            type='bind',
            read_only=False             
        )
    ],
        docker_conn_id="docker_default",
        network_mode="bridge",
        auto_remove='force',
        do_xcom_push=False,
    )

    
    predict_next_signal_task = DockerOperator(
        task_id="predict_latest_candle",
        image=DOCKER_IMAGE_NAME,
        command="python run_pipeline.py predict", 
        mounts=[
        Mount(
            source='/opt/airflow/models',   
            target='/app/models',           
            type='bind',                  
            read_only=False                
        )
    ],
        docker_conn_id="docker_default",
        network_mode="bridge",
        auto_remove='force',
    )

    train_model_task >> predict_next_signal_task

