from .dependencies import *
from .anchor import Anchor


class Database:
    def __init__(self, host, user, password, database_name):
        self.host = host
        self.user = user
        self.password = password
        self.database_name = database_name
        self.database = None
        self.database_cursor = None

    def establish_connection(self):
        self.database = mysql.connector.connect(host=self.host, user=self.user, passwd=self.password,
                                                database=self.database_name)
        self.database_cursor = self.database.cursor()

    def update_certainty_coefficient(self, certainty_coef):
        """
        updates that mysql server with the current certainty coef
        """

        sql = "UPDATE possible_locations SET certainty_coef = %s"
        val = (certainty_coef * 100,)
        self.database_cursor.execute(sql, val)
        self.database.commit()
        # print(mycursor.rowcount, "certainty coef updated .")
        return certainty_coef

    def update_location_estimation(self, location_string):
        """
        updates the estimated location of the bus between two anchors
        """

        sql = "UPDATE possible_locations SET location_estimation = %s"
        val = (location_string,)
        self.database_cursor.execute(sql, val)
        self.database.commit()  # print(mycursor.rowcount, "Location estimate updated .")

    def load_anchor_map(self, WEIGHT):
        """
        Loading anchor map from database
        """
        anchor_map = []
        sql = "SELECT * FROM anchors WHERE weight > %s"
        val = (WEIGHT,)
        self.database_cursor = self.database.cursor()
        self.database_cursor.execute(sql, val)
        myresult = self.database_cursor.fetchall()

        for anchorList in myresult:
            anchor = Anchor(anchorList)
            anchor_map.append(anchor)

        return anchor_map

    def update_possible_locations(self, locations_string):
        sql = "UPDATE possible_locations SET locations_string = %s"
        val = (locations_string,)
        self.database_cursor.execute(sql, val)
        self.database.commit()  # print(mycursor.rowcount, "postion inserted.")

    def initialize_possible_locations(self, anchor_type):
        """
        Get all possible locations from database after detecting first anchor
        """
        possible_locations = []
        self.database_cursor = self.database.cursor()
        sql = "SELECT * FROM anchors WHERE anchor_type = %s and weight > 10"
        self.database_cursor.execute(sql, (anchor_type,))
        results = self.database_cursor.fetchall()
        for anchorList in results:
            anchor = Anchor(anchorList)
            possible_locations.append(anchor)

        return possible_locations

    def save_anchor(self, anchorType, distanceFromPrev, latitude, longitude, heading, weight):
        sql = "INSERT INTO anchors (anchor_type,distance_from_prev,lat,lng,heading,weight) VALUES (%s, %s,%s,%s, %s,%s)"
        sql=""
        val = (anchorType, distanceFromPrev, latitude, longitude, heading, weight)
        self.database_cursor.execute(sql, val)
        self.database.commit()
        print(self.database_cursor.rowcount, "anchor saved.")

    def save_alternate_anchor(self, Route, Segment, anchorType, latitude, longitude, heading, distance):
        sql = "INSERT INTO alternates (route, segment,anchor_type,latitude,longitude,heading,distance) VALUES (%s, %s,%s,%s, %s,%s,%s)"
        val = (Route, Segment, anchorType, latitude, longitude, heading, distance)
        self.database_cursor.execute(sql, val)
        self.database.commit()
        print(self.database_cursor.rowcount, "anchor saved at distance :", distance)

    def get_max_id(self):
        self.database_cursor = self.database.cursor()
        self.database_cursor.execute("SELECT MAX(id) FROM anchors")
        results = self.database_cursor.fetchall()
        return results[0]

    def update_weight(self, the_anchor, type):
        old_weight = the_anchor.weight
        if type == '+':
            new_weight = 15 if old_weight >= 14 else old_weight + 1
        else:
            new_weight = 0 if old_weight < 1 else old_weight - 1
        sql = "UPDATE anchors SET weight = %s WHERE id = %s"
        val = (new_weight, the_anchor.id)
        self.database_cursor.execute(sql, val)
        self.database.commit()  # print(mycursor.rowcount, "postion inserted.")



