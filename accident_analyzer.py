accident_starting_frame = []
accident_ending_frame = []
accident_range_li = []
accident_area_li = []
accident_object_li = []

def analyze():
    accident_cnt = 5
    accident_starting_frame = [1, 1, 1, 1, 1]
    accident_ending_frame = [1, 1, 1, 1, 1]
    x = [255.0, 255.0, 255.0, 255.0, 255.0]
    y = [255.0, 255.0, 255.0, 255.0, 255.0]
    z = [255.0, 255.0, 255.0, 255.0, 255.0]
    radius = [90, 90, 90, 90, 90]

    accident_objects_set = {"AC000000":["3Dd001","3Dd001"], "AC000001": ["3Dd001","3Dd001"]}

    for i in range(accident_cnt):
        accident_range_li.append([accident_starting_frame[i], accident_ending_frame[i]])

    for i in range(accident_cnt):
        accident_area_li.append([x[i], y[i], z[i], radius[i]])
    for accident_ids, object_ids in accident_objects_set.items():
        accident_object_li.append([accident_ids,object_ids])

def accident_range():
    return accident_range_li

def accident_area():
    return accident_area_li

def accident_object():
    print(accident_object_li)
    return accident_object_li

if __name__ == '__main__':
    analyze()
    accident_range()
    accident_area()
    accident_object()


