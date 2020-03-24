from ..common.dependencies import *


class Anchor:
    def __init__(self, anchorList):
        self.id = anchorList[0]
        self.anchor_type = anchorList[1]
        self.lat = anchorList[2]
        self.lng = anchorList[3]
        self.heading = anchorList[4]
        self.distance_from_prev = anchorList[5]
        self.weight = anchorList[6]

    def get_next_anchor(anchor_map, anchor_id):
        """
        Knowing the Id of the anchor , get its location in the
        anchor_map array
        """

        def get_indexes(anchor_id, anchor_map):
            return [theAnchorIndex for (
                theAnchor, theAnchorIndex) in zip(
                anchor_map, range(
                    len(anchor_map))) if theAnchor.id == anchor_id]

        # print(get_indexes(anchor_id,anchor_map))
        return get_indexes(anchor_id, anchor_map)[0]

    def get_distance_to_another_anchor(self, lati2, long2):
        R = 6356.8  # radius of earth
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lng)
        lat2 = math.radians(lati2)
        lon2 = math.radians(long2)

        delta_lat = math.radians(lati2 - self.lat)
        delta_long = math.radians(self.lng - long2)

        haversine = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1) * math.cos(lat2) * math.sin(
            delta_long / 2) * math.sin(delta_long / 2)
        distance_in_between = 2 * R * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))

        return distance_in_between

    def get_bearing_to_another_anchor(self, the_other_anchor):
        R = 6356.8  # radius of earth
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lng)
        lat2 = math.radians(the_other_anchor.lat)
        lon2 = math.radians(the_other_anchor.lng)

        delta_long = math.radians(self.lng - the_other_anchor.lng)

        X = math.cos(lat2) * math.sin(delta_long)
        Y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_long)

        bearing = math.atan2(X, Y)

        return bearing

    def Load_nearby_anchors(self, anchor_map, radius):
        """
        Get all possible locations from database after detecting first anchor
        """
        nearby_anchors = []

        for One_anchor in anchor_map:
            distance = self.get_distance_to_another_anchor(One_anchor.lat, One_anchor.lng)
            if (distance < radius):
                nearby_anchors.append(One_anchor)

        return nearby_anchors

    def calculate_nearby_bearings(self, nearby_anchors):
        bearings = []

        for One_anchor in nearby_anchors:
            bearing = self.get_bearing_to_another_anchor(One_anchor) * 180 / math.pi
            bearing = (360 - ((bearing + 360) % 360))
            bearings.append(bearing)

        return bearings

    def calculate_nearby_distances(self, nearby_anchors):
        distances = []

        for One_anchor in nearby_anchors:
            distance = self.get_distance_to_another_anchor(One_anchor.lat, One_anchor.lng)
            distances.append(distance)

        return distances

    def detection_anomaly_type(self, expected_anchor, detected_anchor):
        distance1 = self.get_distance_to_another_anchor(expected_anchor.lat, expected_anchor.lng)
        distance2 = self.get_distance_to_another_anchor(detected_anchor.lat, detected_anchor.lng)
        if (distance1 > distance2):
            # new anchor detected
            anomaly_type = 'new anchor'
        else:
            # anchor obstructed
            anomaly_type = 'obstructed'
        return anomaly_type

    def anchor_exists_in_db(self, anchor_map):
        for one_anchor in anchor_map:
            distance = self.get_distance_to_another_anchor(one_anchor.lat, one_anchor.lng)
            if distance < 0.01:
                return one_anchor
        return False

    def recalibrate(self, anchor_map,radius):
        possible_locations=[]
        for the_anchor in anchor_map:
            distance = self.get_distance_to_another_anchor(the_anchor.lat, the_anchor.lng)
            if distance < radius:
                possible_locations.append(the_anchor)

        return possible_locations



