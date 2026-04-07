"""Tests for momapy_kb utility functions."""

import pytest

import momapy_kb.utils


class TestFlattenCollection:
    """Tests for flatten_collection function."""

    def test_flat_list(self):
        assert momapy_kb.utils.flatten_collection([1, 2, 3]) == [1, 2, 3]

    def test_nested_lists(self):
        assert momapy_kb.utils.flatten_collection([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_deeply_nested(self):
        assert momapy_kb.utils.flatten_collection([[[1]], [[2]]]) == [1, 2]

    def test_mixed_nesting(self):
        assert momapy_kb.utils.flatten_collection([1, [2, [3]], 4]) == [1, 2, 3, 4]

    def test_strings_not_flattened(self):
        result = momapy_kb.utils.flatten_collection(["abc", "def"])
        assert result == ["abc", "def"]

    def test_empty_list(self):
        assert momapy_kb.utils.flatten_collection([]) == []

    def test_single_element(self):
        assert momapy_kb.utils.flatten_collection([42]) == [42]


class TestCoreDataModels:
    """Tests for CollectionEntry and Collection dataclasses."""

    def test_collection_entry_creation(self):
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        entry = momapy_kb.core.CollectionEntry(
            id_="test",
            model=SimpleModel(name="test_model"),
        )
        assert entry.id_ == "test"
        assert entry.file_path is None
        assert entry.rdf_annotations is None
        assert entry.ids is None
        assert entry.notes is None

    def test_collection_entry_is_frozen(self):
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        entry = momapy_kb.core.CollectionEntry(
            id_="test",
            model=SimpleModel(name="test"),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.id_ = "changed"

    def test_collection_creation(self):
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class SimpleModel:
            name: str

        entry = momapy_kb.core.CollectionEntry(
            id_="test",
            model=SimpleModel(name="test"),
        )
        collection = momapy_kb.core.Collection(
            name="TestCollection",
            entries=frozenset([entry]),
        )
        assert collection.name == "TestCollection"
        assert len(collection.entries) == 1

    def test_collection_default_empty_entries(self):
        collection = momapy_kb.core.Collection(name="Empty")
        assert collection.entries == frozenset()

    def test_collection_is_frozen(self):
        import dataclasses

        collection = momapy_kb.core.Collection(name="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            collection.name = "changed"


import momapy_kb.core


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
