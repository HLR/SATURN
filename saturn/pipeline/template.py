"""Code template — the scaffold that wraps generated spatial programs."""


CODE_TEMPLATE = """
def logic_executor(query, score_fn, query_fn, scene, images, history):
    score = lambda x, num_objects=1, type=None, cam_id=None: score_fn(
        question=x,
        num_objects=num_objects,
        type=type,
        cam_id=cam_id,
        scene=scene,
        images=images,
    )
    query = lambda x, object_id=None, type="default": query_fn(
        question=x,
        object_id=object_id,
        type=type,
        scene=scene,
        images=images,
    )
    objects_count = scene.objects_count
    camera = formula_helpers(score, scene)["camera"]
    {code}
"""
