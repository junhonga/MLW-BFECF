
class Dispatcher():

    def __init__(self, Template):
        self.Template = Template

    def obtain_instance(self, config):
        est_name = config.get("Name", None)
        est_type = config.get("Type", None)
        return self._obtain_instance(est_name, est_type, config)

    def _obtain_instance(self, name, est_type, config):
        est = self.execute_dispatcher_method(name, est_type, config)
        if isinstance(est, self.Template):
            return est
        else:
            raise est.name + "没有继承 " + self.Template + " 类"

    def execute_dispatcher_method(self, name, est_type, configs):
        pass

class ListDispatcher(Dispatcher):

    def obtain_instance(self, configs):
        multi_ests = []
        for name, config in configs.items():
            est_type = config.get("Type", None)
            est = super()._obtain_instance(name, est_type, config)
            multi_ests.append(est)
        return multi_ests

class MultiDispatcher(Dispatcher):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def obtain_instance(self, configs):
        multi_ests = {}
        for m_name, config in configs.items():
            multi_ests[m_name] = self.dispatcher().obtain_instance(config)
        return multi_ests