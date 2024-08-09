from copy import deepcopy
from torch import nn

from util.context import context


class SceneModule(nn.Module):
    """ Manages access to context and scene"""

    def __init__(self):
        super().__init__()
        assert hasattr(context, "scene"), f"Can only initialize {self.__class__.__name__} inside 'with context(scene=...):'"
        self._scene = [context.scene]

    def __deepcopy__(self, memo):
        """
        The "scene" of the copied object is based on the context.
        - If context.scene is True:    perform a regular deepcopy (used when copying a scene wholesale)
        - If context.scene is False:   new object has no scene attached (used when copying only party of a scene for saving/caching)
        - If context.scene is a scene: new object has the context.scene attached (used when copying from one scene into another)
        """
        assert hasattr(context, "scene"), f"Can only deepcopy {self.__class__.__name__} inside 'with context(scene=...):'"

        if context.scene is True:
            self.__deepcopy__ = None
            x = deepcopy(self, memo)
            del self.__deepcopy__
            del x.__deepcopy__

        else:
            s = self._scene
            del self._scene

            self.__deepcopy__ = None
            x = deepcopy(self, memo)
            del self.__deepcopy__
            del x.__deepcopy__

            self._scene = s

            if context.scene is False:
                x._scene = [None]
            else:
                x._scene = [context.scene]

        return x
        
    @property
    def scene(self):
        return self._scene[0]

    @property
    def audio_sr(self):
        return self.scene.audio_sr

    @property
    def scene_duration(self):
        return self.scene.scene_duration