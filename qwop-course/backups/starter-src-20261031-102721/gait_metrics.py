"""课程验收标准：不随候选模型的结果改动。"""
def walking_pass(distance, nonfoot, knee_low, upright, switches, target=10.0):
    return bool(distance >= target and nonfoot <= .05 and knee_low <= .05
                and upright >= .8 and switches >= 4)
