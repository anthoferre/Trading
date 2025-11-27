from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta
from docker.types import Mount

# --- CONFIGURATION GLOBALE ---
DOCKER_IMAGE_NAME = "ghcr.io/anthoferre/trading:latest"

with DAG(
    dag_id="hourly_trading_prediction",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="1 * * * 1-5", 
    catchup=False,
    default_args={
        "owner": "airflow",
        "email": ["anthony.ferre@hotmail.fr"],
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["trading", "prediction", "hourly"],
) as dag:

    # 1. Tâche de Prediction (Hourly)
    predict_next_signal_task = DockerOperator(
        task_id="predict_latest_candle",
        image=DOCKER_IMAGE_NAME,
        command="predict", 
        mounts=[
        Mount(
            source='/opt/airflow/models',   
            target='/app/models',           
            type='bind',                  
            read_only=False                
        )
    ],
        docker_conn_id="docker_default", 
        auto_remove='force',
    )

    
    # --- DEFINITION DU FLUX DE TRAVAIL ---
    predict_next_signal_task