# torch 의존 모듈은 lazy import — torch 없는 환경에서도 서버 기동 가능
__all__ = [
    "TinyTransformer", "STMMT", "OceanCubeDataset",
    "SparseMHA", "DynamicFFN", "SensorEmbedding", "PatchEmbedding",
    "TokenImportanceScorer", "CrossModalFusion",
]


def __getattr__(name):
    if name in __all__:
        import importlib
        mapping = {
            "TinyTransformer":       ("ml.models.tiny_transformer", "TinyTransformer"),
            "STMMT":                 ("ml.models.st_mmt", "STMMT"),
            "OceanCubeDataset":      ("ml.models.st_mmt", "OceanCubeDataset"),
            "SparseMHA":             ("ml.models.sparse_attention", "SparseMHA"),
            "DynamicFFN":            ("ml.models.dynamic_ffn", "DynamicFFN"),
            "SensorEmbedding":       ("ml.models.embedding", "SensorEmbedding"),
            "PatchEmbedding":        ("ml.models.embedding", "PatchEmbedding"),
            "TokenImportanceScorer": ("ml.models.token_scorer", "TokenImportanceScorer"),
            "CrossModalFusion":      ("ml.models.cross_modal", "CrossModalFusion"),
        }
        mod_path, attr = mapping[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'ml.models' has no attribute {name!r}")
