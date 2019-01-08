import textworld
import sys,os

game_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "zork1.z5"))

class ZorkEnv(object):
    """
    An environment wrapper for TextWorld custom environment:Zork.
    Provides state extraction and helper functions.
    """
    def __init__(self):
        self.env = None

    def start(self):
        self.env = textworld.start(game_file)
        return self.env
