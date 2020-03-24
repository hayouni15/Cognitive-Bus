class target:
    def __init__(self, x_center, y_center, width, height):
        self.x_center = x_center
        self.y_center = y_center
        self.height = height
        self.width = width

    def exists(self, car):
        centerx = self.x_center
        centery = self.y_center

        if (centerx < car.x_center + car.width/2) and ((centerx > car.x_center - car.width/2 )) and (
                centery < car.y_center + car.height/2 ) and (centery > car.y_center - car.height/2):
            exist = True
        else:
            exist = False
        return exist
