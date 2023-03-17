import configparser


class Config:

    def __init__(self, config_file: str):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.tool = config["main"]["tool"]
        self.url = config["main"]["url"]
        self.username = config["admin"]["username"]
        self.password = config["admin"]["password"]

    def __repr__(self):
        return f"tool = {self.tool} | " \
               f"url = {self.url} | " \
               f"username = *** | " \
               f"password = ***"
