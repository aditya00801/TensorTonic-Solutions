def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    rows = len(data)
    cols = len(data[0])

    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        minimum  = min(data[i][j] for i in range(rows))
        maximum = max(data[i][j] for i in range(rows))

        for i in range(rows):
            if maximum == minimum:
                result[i][j] = 0.0
            else:
                result[i][j] =(
                    (data[i][j] - minimum) / (maximum - minimum)
                )


    return result 