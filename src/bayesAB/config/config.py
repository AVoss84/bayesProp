from my_package.services.file import YAMLService

my_yaml = YAMLService(path = "my_package/config/input_output.yaml")
io = my_yaml.doRead()

my_yaml2 = YAMLService(path = "my_package/config/model_config.yaml")
model_list = my_yaml2.doRead()
