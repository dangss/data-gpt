import requests
import time

_AIRFLOW_BASE_URL = 'https://autoeda-airflow.di.zalo.services/api/v1'
_HEADER = {'Content-Type': 'application/json',
        'Authorization': 'Basic YWlyZmxvdzphaXJmbG93'
        }


def create_session_no_proxies():
    proxies = {
    'http': '',
    'https': ''
    }
    session = requests.Session()
    session.trust_env = False
    session.proxies.update(proxies)
    return session


def get_dag_run_status(dag_run_id, dag_name):
    url = f'{_AIRFLOW_BASE_URL}/dags/{dag_name}/dagRuns/{dag_run_id}'
    response = create_session_no_proxies().get(url=url, headers=_HEADER)
    n_retries = 0
    while ((response is None) or 
           (('conf' in response.json()) and (response.json()['state'] is None))
           ) and n_retries < 3:
        print(f'Airflow webserver does not response. Up for retry {n_retries+1}')
        time.sleep(2)
        n_retries += 1
        response = create_session_no_proxies().get(url=url, headers=_HEADER)
    if response is None:
        return None
    if 'conf' not in response.json():
        return None
    return response.json()['state']


def airflow_trigger_dag(dag_name, conf):
    url = f'{_AIRFLOW_BASE_URL}/dags/{dag_name}/dagRuns'
    response = create_session_no_proxies().post(url=url, json=conf, headers=_HEADER)
    if response.status_code != 200:
        return False, response
    return True, response