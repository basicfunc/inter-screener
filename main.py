from dotenv import dotenv_values

config = dotenv_values('.env')
print(config.keys())
print(config.values())