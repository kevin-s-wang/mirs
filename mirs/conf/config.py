
import os
import sys
from posixpath import join as urljoin
from pathlib import Path
from typing import Dict
import yaml

ROOT_DIR = Path(os.path.dirname(__file__)).parent.parent.resolve()
DEFAULT_CONFIG_FILENAME = 'mirs.yaml'

class ClipConfiguration:
    def __init__(self, config: Dict) -> None:
        self._config = config

    def get_model(self) -> str:
        return self._config['model']
    
    def get_pretrained(self) -> str:
        return self._config['pretrained']
    
    def get_embeddings_dim(self) -> int:
        return self._config['embeddings_dim']

class ServerConfiguration:

    def __init__(self, config: Dict) -> None:
        self._config = config
    
    def get_port(self) -> int:
        return self._config['port']
    
    def get_host(self) -> str:
        return self._config['host']
    
    def get_relative_images_dir(self) -> str:
        return self._config['relative_images_dir']
    
    def get_images_url(self, filepath: str) -> str:
        return urljoin(self._config['images_url_base'], filepath)
    
    def get_embeddings_api_url(self) -> str:
        return self._config['endpoints']['embeddings']
    
    def get_images_api_url(self) -> str:
        return self._config['endpoints']['images']
    
    def get_tags_api_url(self, image_id: str) -> str:
        return self.get_images_api_url() + f'/{image_id}/tags'


class Configuration:

    @property
    def root_dir(self) -> str:
        return self._config.get('root_dir')
    
    @property
    def dev_dir(self) -> str:
        return self._config.get('dev_dir')

    @property
    def datasets_dir(self) -> str:
        return self._config.get('datasets_dir')
    
    def get_images_dir(self) -> str:
        return os.path.join(ROOT_DIR, self.get_relative_images_dir())
    
    def get_relative_images_dir(self) -> str:
        return self._config['server']['relative_images_dir']

    def get_images_url(self, filepath: str) -> str:
        return urljoin(self._config['server']['images_url_base'], filepath)
    
    def get_embeddings_api_url(self) -> str:
        return self._config['server']['endpoints']['embeddings']
    
    def get_images_api_url(self) -> str:
        return self._config['server']['endpoints']['images']
    
    def get_server_host(self) -> str:
        return self._config['server']['host']

    def get_server_port(self) -> int:
        return self._config['server']['port']

    def get_database_uri(self) -> str:
        return self._config['server']['database']['uri']
    
    def get_kafka_bootstrap_servers(self) -> str:
        return self._config['server']['kafka']['bootstrap_servers']

    def get_as_str(self, key: str, default=None) -> str:
        return self._config.get(key, default)
    
    def get_log_level(self) -> str:
        return self._config['server']['logging']['level'] or 'info'

    def __init__(self, config_path: str = None) -> None:
        self.config_path: str = config_path or os.path.join(ROOT_DIR, DEFAULT_CONFIG_FILENAME)
        self._config = {}
        self._load_config()
        
        self.server = ServerConfiguration(self._config['server'])
        self.clip = ClipConfiguration(self._config['clip'])
        self.llm = LLMConfiguration(self._config['llm'])

    def _load_config(self):
        with open(self.config_path, 'r') as config_file:
            self._config: Dict = yaml.safe_load(config_file)
        self._config['root_dir'] = ROOT_DIR
        self._config['dev_dir'] = os.path.join(ROOT_DIR, '.dev')
        self._config['datasets_dir'] = os.path.join(self.dev_dir, 'datasets')


class LLMConfiguration:

    def __init__(self, config: Dict) -> None:
        self._config = config

    def get_service_class(self) -> str:
        return self._config['service_class']

config = Configuration()