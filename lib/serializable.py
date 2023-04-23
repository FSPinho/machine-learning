class Serializable:
    def serialize(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def deserialize(instance_dict: dict):
        raise NotImplementedError
