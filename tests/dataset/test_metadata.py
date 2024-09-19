def test_get_metadata_from_dataset(dataset) -> None:
    dataset.add_metadata("something", 123)
    something = dataset.get_metadata("something")
    assert 123 == something


def test_get_nonexisting_metadata(dataset) -> None:
    data = dataset.get_metadata("something")
    assert data is None


def test_get_metadata_lower_upper_case(dataset) -> None:
    dataset.add_metadata("something", 123)

    something_lower = dataset.metadata.get("something", "didnt find lowercase")
    assert something_lower == 123

    something_upper = dataset.metadata.get("Something", "didnt find uppercase")
    assert something_upper == "didnt find uppercase"

    get_something_lower = dataset.get_metadata("something")
    assert get_something_lower == 123

    get_something_upper = dataset.get_metadata("Something")
    assert get_something_upper is None
