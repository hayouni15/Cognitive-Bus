from ..common.dependencies import *


class tracker:
    def __init__(self, track_points, track_count, tracked_count):
        self.track_points = track_points
        self.track_count = track_count
        self.tracked_count = tracked_count
        self.track_color = self.get_random_color()
        self.sum=0


    def get_random_color(self):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return (red, green, blue)

    def draw_track(self, frame):
        points_limit = 50
        points_counter = 0
        cv2.putText(frame, str(int(self.sum))+' m', (int(self.track_points[len(self.track_points)-1].x_center), int(self.track_points[len(self.track_points)-1].y_center)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
        for coordinates in reversed(self.track_points):

            points_counter += 1
            if points_counter < points_limit:
                cv2.circle(frame, (int(coordinates.x_center), int(coordinates.y_center)), 3, self.track_color, -1)


    def estimate_coef(self):
        x = []
        y = []
        points_limit_count = 0
        for coordinates in self.track_points:
            x.append(int(coordinates.x_center))
            y.append(int(coordinates.y_center))
        x = np.array(x)
        y = np.array(y)
        if len(x) > 51:
            x = x[-50:]
            y = y[-50:]
        # number of points
        n = np.size(x)

        # mean of x and y vector
        m_x, m_y = np.mean(x), np.mean(y)

        # calculating cross-derivation and deviation about x

        SS_xy = np.sum(y * x) - n * m_y * m_x
        SS_xx = np.sum(x * x) - n * m_x * m_x

        # calculate regression coef
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1 * m_x

        return (b_0, b_1)

    def plot_regression_vector(self, frame):
        b = self.estimate_coef()
        final_point_x = self.track_points[len(self.track_points) - 1].x_center
        if len(self.track_points) < 10:
            initial_point_x = self.track_points[0].x_center
        else:
            initial_point_x = self.track_points[len(self.track_points) - 9].x_center

        x_diff = initial_point_x - final_point_x
        initial_point_x = final_point_x - x_diff

        initial_point_y = b[0] + b[1] * initial_point_x
        final_point_y = b[0] + b[1] * final_point_x
        if math.isnan(initial_point_y) and math.isnan(final_point_y):
            print('nan case')
        else:
            cv2.arrowedLine(frame, (int(final_point_x), int(final_point_y)),
                            (int(initial_point_x), int(initial_point_y)), (51, 255, 255), 3)

    def plot_extended_line(self, frame, object, warning_counter):
        return_counter_increment = 0
        b = self.estimate_coef()
        if len(self.track_points) > 20:
            initial_point_x = self.track_points[len(self.track_points) - 21].x_center
        else:
            initial_point_x = self.track_points[0].x_center
        initial_point_y = b[0] + b[1] * initial_point_x

        last_point_x = self.track_points[len(self.track_points) - 1].x_center
        last_point_y = b[0] + b[1] * last_point_x

        vector_magnitude = math.sqrt((last_point_x - initial_point_x) ** 2 + (last_point_y - initial_point_y) ** 2)

        if last_point_y < initial_point_y:
            final_point_y = 0
        else:
            final_point_y = frame.shape[0]
            final_point_x = (final_point_y - b[0]) / b[1]

            y_test = frame.shape[0]
            x_test = (y_test - b[0]) / b[1]
            x_center = frame.shape[1] / 2

            if math.isnan(initial_point_y) or math.isnan(final_point_y) or final_point_x == float(
                    "inf") or final_point_x == float("-inf") or abs(
                final_point_x) > 2147483648 or vector_magnitude < 20:
                print('nan case')
            else:
                if x_test < x_center + 80 and x_test > x_center - 80:
                    cv2.line(frame, (int(last_point_x), int(last_point_y)),
                             (int(final_point_x), int(final_point_y)), (0, 0, 204), 2)
                    cv2.circle(frame, (int(frame.shape[1] / 2), frame.shape[0] + 50), 100, (0, 0, 204), 3)
                    return_counter_increment = 1
                    if warning_counter > 5:
                        cv2.rectangle(frame, (5, 5), (1910, 1070), (0, 0, 204), 5)
                        cv2.rectangle(frame, (600, 15), (1300, 80), (0, 0, 204), -1)
                        cv2.putText(frame, "Collision warning: " + object + " ahead Slow down", (620, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)

                elif x_test < x_center + 150 and x_test > x_center - 150:
                    cv2.line(frame, (int(last_point_x), int(last_point_y)),
                             (int(final_point_x), int(final_point_y)), (51, 153, 255), 2)
                    cv2.circle(frame, (int(frame.shape[1] / 2), frame.shape[0] + 125), 200, (51, 153, 255), 3)

                    if warning_counter > 5:
                        cv2.rectangle(frame, (5, 5), (1910, 1070), (0, 0, 204), 5)
                        cv2.rectangle(frame, (600, 15), (1300, 80), (0, 0, 204), -1)
                        cv2.putText(frame, "Collision warning: " + object + " ahead Slow down", (620, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
                    return_counter_increment = 1

                elif x_test < x_center + 220 and x_test > x_center - 220:
                    cv2.line(frame, (int(last_point_x), int(last_point_y)),
                             (int(final_point_x), int(final_point_y)), (51, 255, 153), 2)
                    cv2.circle(frame, (int(frame.shape[1] / 2), frame.shape[0] + 200), 300, (51, 255, 153), 3)

                    if warning_counter > 5:
                        cv2.rectangle(frame, (5, 5), (1910, 1070), (0, 0, 204), 5)
                        cv2.rectangle(frame, (600, 15), (1300, 80), (0, 0, 204), -1)
                        cv2.putText(frame, "Collision warning: " + object + " ahead Slow down", (620, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
                    return_counter_increment = 1

                else:
                    cv2.line(frame, (int(last_point_x), int(last_point_y)),
                             (int(final_point_x), int(final_point_y)), (128, 128, 128), 1)
                    return_counter_increment = 0

        return return_counter_increment

    def depth_distance(self,depth_map):
        x_center=self.track_points[len(self.track_points)-1].x_center
        y_center=self.track_points[len(self.track_points)-1].y_center
        box_height=self.track_points[len(self.track_points)-1].height
        box_width=self.track_points[len(self.track_points)-1].width
        cropped=depth_map[int(y_center-box_height/2):int(y_center+box_height/2),int(x_center-box_width/2):int(x_center+box_width/2)]
        #cv2.imshow('cropped',cropped)
        print('sum ',np.sum(cropped))
        x=np.sum(cropped) / (box_height * box_width)
        self.sum=99.2435-0.700477*x+0.001363*x*x