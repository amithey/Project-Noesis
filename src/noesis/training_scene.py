"""Quiet ground and a visible physical obstacle; no geometric neural drive."""
import numpy as np

from noesis.habitat import Habitat


class QuietFields(Habitat):
    patches = ()

    def odors_at(self, points, time_s=0.):
        return np.zeros((6, len(points)))

    def tastes_at(self, points):
        return np.zeros((len(points), 3))

    def local_conditions(self, position, time_s):
        return {'temperature_c': 24., 'humidity': .45, 'sound_envelope': 0.,
                    'airflow_envelope': 0., 'illumination': 1.}


def training_arena(obstacle=False):
    from flygym.arena import FlatTerrain

    class TrainingArena(FlatTerrain):
        habitat = QuietFields()

        @property
        def odor_dimensions(self):
            return 6

        def get_olfaction(self, sensor_pos):
            return self.habitat.odors_at(sensor_pos)

    arena = TrainingArena()
    arena.root_element.find('geom', 'ground').pos = (0, 0, 0)
    arena.root_element.find('material', 'grid').reflectance = 0
    if obstacle:
        arena.root_element.worldbody.add('geom', name='obstacle', type='box',
            pos=(2.5, 0, .75), size=(.2, 2., .75), rgba=(.3, .2, .1, 1.))
    return arena
