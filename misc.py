class Json2Obj:
    def __init__(self, data):
        self.__dict__ = data
        for i in self.__dict__.keys():
            child = self.__dict__[i]
            if isinstance(child, dict):
                if len(child) > 0:
                    self.__dict__[i] = Json2Obj(child)
            if isinstance(child, list):
                self.__dict__[i] = []
                for item in child:
                    if isinstance(item, dict):
                        self.__dict__[i].append(Json2Obj(item))
                    else:
                        self.__dict__[i].append(item)
