class Debuggable:
    def __init__(self):
        self._debug = False

    def enable_debug(self):
        self._debug = True
        return self

    def disable_debug(self):
        self._debug = False
        return self

    def _debug_log(self, *args, **kwargs):
        if self._debug:
            print(*[f"{self.__class__.__name__}:", *self._prepare_value_to_log(args)], **kwargs)

    def _prepare_value_to_log(self, value):
        if isinstance(value, tuple):
            return map(self._prepare_value_to_log, value)

        if isinstance(value, list):
            return f"[ {', '.join(map(self._prepare_value_to_log, value))} ]"

        if isinstance(value, (int, float)):
            return "{value:.8f}".format(value=value)

        return value
