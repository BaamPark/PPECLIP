import yaml

#single ton class
class Config:
    _instance = None  # Stores the singleton instance of Config
    _config_file = None

    def __new__(cls):
        #Checks if _instance is None. If no instance exists, create one
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    @classmethod #allows methods to access class variables (_config_file).
    def set_config_file(cls, file_path):
        cls._config_file = file_path
        
    def load_config(self):
        with open(self._config_file, "r") as file:
            self.cfg = yaml.safe_load(file)  # Load YAML into a dictionary

def get_config():
    return Config().cfg  # Access singleton instance and return config data
