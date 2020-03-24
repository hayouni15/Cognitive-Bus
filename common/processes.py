from ..common.dependencies import *

def process_detection(pred1, pred2):
    """
    and outputs an array of detected elements at each frame without duplication
    """
    pred = []
    exists = False
    for p1 in pred1:
        for p in pred:
            if p[0] == p1[0]:
                exists = True
        if not exists:
            if (p1[0] in [b'traffic light', b'fire hydrant', b'stop sign', b'benchXXX', b'parking meterXXX']):
                pred.append(p1)
        exists = False

    exists = False
    for p2 in pred2:
        for p in pred:
            if p[0] == p2[0]:
                exists = True
        if not exists:
            if (p2[0] in [b'traffic light', b'fire hydrant', b'stop sign', b'benchXXX', b'parking meterXXX']):
                pred.append(p2)
        exists = False
    return pred
def process_prediction(pred1, pred2):
    """
    this function takes as input the output of predictor one and predictor two
    and outputs an array of detected elements at each frame without duplication
    """
    pred = []
    exists = False
    for p1 in pred1:
        for p in pred:
            if p[0] == p1[0]:
                exists = True
        if not exists:
            if (p1[0] in [b'traffic light', b'fire hydrant', b'stop sign', b'benchXXX', b'parking meterXXX']):
                pred.append(p1)
        exists = False

    exists = False
    for p2 in pred2:
        for p in pred:
            if p[0] == p2[0]:
                exists = True
        if not exists:
            if (p2[0] in [b'traffic light', b'fire hydrant', b'stop sign', b'benchXXX', b'parking meterXXX']):
                pred.append(p2)
        exists = False
    return pred

def assert_detection(TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter, prediction, ismoving=True):
    """
    this function will keep track of the detection count
    it will increment each anchor count if detection countinues
    and restart counter if detection interrupted
    """
    out = [0.6, 0.7, 0.9, 0.7, 0.7]
    for p in prediction:
        if p[0] == b'traffic light':
            out[0] = 1
            TL_counter += 1.2 * p[1]
        if p[0] == b'fire hydrant':
            out[1] = 1
            FH_Counter += 2.5 * p[1]
        if p[0] == b'stop sign':
            out[2] = 1
            TS1_Counter += 2.9 * p[1]
        if p[0] == b'bench':
            out[3] = 1
            TS2_Counter += 2
        if p[0] == b'parking meter':
            out[4] = 1
            TS3_Counter += 1
    counters = [TL_counter, FH_Counter, TS1_Counter, TS2_Counter, TS3_Counter]
    out = np.multiply(out, counters)
    return out[0], out[1], out[2], out[3], out[4]



def max_detection_count(current, max):
    for i in range(0, len(current)):
        if current[i] > max[i]:
            max[i] = current[i]
    return max