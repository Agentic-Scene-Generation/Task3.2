from scenesmith.scenebenchmark_critic.metrics.functional_dependency import relations


def test_non_support_relation_never_reverses_into_support(monkeypatch) -> None:
    shelf = {"id": "bookshelf_0", "category_norm": "bookshelf"}
    book = {"id": "book_0", "category_norm": "book"}

    monkeypatch.setattr(
        relations,
        "_relation_target_is_valid",
        lambda subject, target, relation: (subject["id"], target["id"], relation)
        == ("book_0", "bookshelf_0", "object_on_support"),
    )
    monkeypatch.setattr(relations, "_infer_relation_type", lambda *_args: None)

    label, _confidence, reason, _diagnostics = relations._eval_relation_over_targets(
        None, shelf, [book], "furniture_faces_furniture"
    )

    assert label == "fail"
    assert "target category is not compatible" in reason
