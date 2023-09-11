def split_data(data):
    """splits data into two halves"""
    midpoint = len(data) // 2
    return data[:midpoint], data[midpoint:]

def sum_data(half_data):
    """sums the data"""
    return sum(half_data)