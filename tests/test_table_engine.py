from ingest.table_engine import TableCell, TableProcessingEngine, TableRegion


def test_semantic_ownership_suppresses_only_text_owned_by_cells():
    region = TableRegion(
        bbox=(0, 0, 100, 100),
        cells=(TableCell("Part", (0, 0, 50, 20)),),
    )
    items = [
        ({"bbox": [1, 1, 20, 10], "text": "Part"}, "text"),
        ({"bbox": [0, 30, 20, 40], "text": "Diagram label"}, "text"),
        ({"bbox": [0, 0, 100, 100], "kind": "image"}, "image"),
    ]

    filtered = TableProcessingEngine().suppress_owned_text(items, [region])

    assert len(filtered) == 2
    assert filtered[0][0]["text"] == "Diagram label"
    assert filtered[1][1] == "image"


def test_vector_table_detection_requires_grid_lines():
    class Page:
        def get_cdrawings(self):
            return [
                {"rect": (0, 0, 100, 1)},
                {"rect": (0, 50, 100, 51)},
                {"rect": (0, 0, 1, 50)},
                {"rect": (50, 0, 51, 50)},
            ]

    regions = TableProcessingEngine().detect_tables(Page())

    assert len(regions) == 1
    assert regions[0].source == "vector_heuristic"
    assert regions[0].bbox == (0.0, 0.0, 100.0, 51.0)
