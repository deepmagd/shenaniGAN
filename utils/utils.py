import yaml

def get_default_settings(settings_file='settings.yml'):
    with open(settings_file) as f:
        return yaml.safe_load(f)
