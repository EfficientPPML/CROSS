"""Complete every cleanup action without replacing primary failures."""


class CleanupTracker:
    def __init__(self, primary):
        self.primary = primary
        self.errors = []
        self.cancellation = None

    def attempt(self, label, action):
        try:
            return action()
        except BaseException as error:
            note = f'{label}: {type(error).__name__}: {error}'
            self.errors.append(note)
            if self.primary is not None:
                self.primary.add_note(note)
            elif self.cancellation is None and not isinstance(error, Exception):
                self.cancellation = error
            return None

    def finish(self):
        if self.primary is None and self.cancellation is not None:
            for note in self.errors:
                self.cancellation.add_note(note)
            raise self.cancellation
