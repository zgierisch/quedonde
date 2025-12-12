def alpha():
    """Alpha function returns a constant."""
    return "alpha"


def beta():
    return alpha()


def gamma():
    value = beta()
    return value.upper()
