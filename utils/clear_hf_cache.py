from huggingface_hub import HfApi
import os
import shutil
from pathlib import Path


def clear_hf_dataset_cache(repo_id: str):
    """Clears the Hugging Face cache for a specific dataset."""

    try:
        # 1. Determine the cache directory:
        cache_dir = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
        assert cache_dir.exists()
        print(cache_dir)

        # 2. List potential cache directories for the given repo_id:
        potential_cache_dirs = [
            d
            for d in cache_dir.iterdir()
            if d.is_dir() and repo_id.replace("/", "--") in d.name
        ]

        if not potential_cache_dirs:
            print(f"No cache directory found for repo_id: {repo_id}")
            return False

        # 3. Iterate and remove potential cache directories
        for cache_dir_to_remove in potential_cache_dirs:
            print(f"Deleting cache directory: {cache_dir_to_remove}")
            shutil.rmtree(cache_dir_to_remove)

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def main():
    repo_id = "mjboothaus/titanic-databooth"
    success = clear_hf_dataset_cache(repo_id)

    if success:
        print(f"Successfully cleared cache for {repo_id}")
    else:
        print(f"Failed to clear cache for {repo_id}")


if __name__ == "__main__":
    main()
