if __name__ == "__main__":
    # execute only if run as a script
    from ex toymodel import AModel, MockGates
    model = AModel()
    gates = MockGates('gates', model=model)