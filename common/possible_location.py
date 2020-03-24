from ..common.dependencies import *
from random import randrange


class Possible_location_stats:
    def __init__(self, original_anchor, nearby_anchor, distance, bearing, bearing_error, probability):
        self.original_anchor = original_anchor
        self.nearby_anchor = nearby_anchor
        self.distance = distance
        self.bearing = bearing
        self.bearing_error = bearing_error
        self.probability = probability
        self.id=randrange(1000)

    def original_anchor_exists(self, stats_array):
        for stat in stats_array:
            if self.original_anchor == stat.original_anchor:
                return True

        return False

    def is_duplicate(self, stats_array):
        for stat in stats_array:
            if (stat.original_anchor == self.original_anchor) and (stat.nearby_anchor == self.nearby_anchor) :
                return True
        return False


