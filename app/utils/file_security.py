import os


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # User-supplied paths may be filenames, relative paths, absolute paths, or contain `../`.
    # Resolve to realpath and verify via commonpath that the target remains within the allowed directory.
    # This is more reliable than prefix checks, properly handling symlinks, duplicate separators,
    # and relative traversals for whitelisted folders (uploads, materials, task artifacts).
    if not unsafe_path:
        raise ValueError("empty path is not allowed")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # Different drive letters on Windows raise ValueError; such paths definitely do not belong to the allowed directory.
        raise ValueError("path is outside the allowed directory") from exc

    if common_path != base_dir_real:
        raise ValueError("path is outside the allowed directory")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("file does not exist")

    return resolved_path
