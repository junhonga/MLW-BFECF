
class ClassifierBuiderPreProcessorTemplate():

    def __init__(self):
        self.is_pre_executable = True
        self.is_post_executable = True

    def set_is_pre_executable(self, is_pre_executable):
        self.is_pre_executable = is_pre_executable

    def obtain_is_pre_executable(self):
        self.is_pre_executable

    def set_is_post_executable(self, is_post_executable):
        self.is_post_executable = is_post_executable

    def obtain_is_post_executable(self):
        self.is_post_executable

    def pre_executable(self, layer):
        return self.is_pre_executable

    def pre_fit_execute(self, data, buider, layer):
        pass

    def post_executable(self, layer):
        return self.is_post_executable

    def post_fit_execute(self, data, buider, layer):
        pass