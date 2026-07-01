"""Basic smoke tests for dataset helper utilities."""

from plantify import data


def test_species_mapping_known_folder():
    assert data.species_of("leaf2") == "Acer"


def test_species_mapping_unknown_folder():
    assert data.species_of("new_species") == "new species"
