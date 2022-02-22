def minkowski_distance(
    data, p, start_lat = 'pickup_latitude', start_lon = 'pickup_longitude', end_lat = 'dropoff_latitude', end_lon = 'dropoff_longitude'
):
    x1 = data[start_lon]
    x2 = data[end_lon]

    y1 = data[start_lat]
    y2 = data[end_lat]
    
    minkowski = ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)

    return minkowski