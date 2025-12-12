import json
from pathlib import Path

# @quedonde:legacy
class Controller:
    def __init__(self) -> None:
        self.state = {}

    def run(self):
        return helper(self.state.get("value", 0))

    @classmethod
    def build(cls):
        return cls()


# @quedonde:bridge
def helper(value):
    Path("/tmp").exists()
    return value * 2
