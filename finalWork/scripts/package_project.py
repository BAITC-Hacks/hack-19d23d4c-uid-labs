"""Package the local project, with supplied data, but never keys or review notes."""
import os
from pathlib import Path
import zipfile


root = Path(__file__).resolve().parents[1]
target = root.parent / "Sled_AI_HackAlem.zip"
count = 0
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory, folders, files in os.walk(root):
        folders[:] = [name for name in folders if name not in
                      {".venv", ".git", "__pycache__", ".pytest_cache", "build", "dist", "node_modules"}
                      and not name.endswith(".egg-info")]
        for name in sorted(files):
            if ((name.startswith(".env") and name != ".env.example")
                    or name.endswith((".pyc", ".zip")) or name == "reviews.jsonl"):
                continue
            source = Path(directory) / name
            archive.write(source, "moneygraph_hackathon/" + source.relative_to(root).as_posix())
            count += 1
print(f"Packaged {count} files: {target} ({target.stat().st_size} bytes)")
