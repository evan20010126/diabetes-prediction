import os

folders = [
    "configs", "cores", "datasets", "launch", "models", "outputs", "scripts",
    "tests", "utils"
]


def create_directory_structure(base_path="."):
    with open(os.path.join(base_path, "setup.py"), 'w') as fp:
        fp.write("""
        # This file is required to make the directory a package.
        from setuptools import setup, find_packages

        setup(
            name='your_project_name',
            version='0.1.0',
            author='Evan',
            packages=find_packages(),
            install_requires=[],
            python_requires='>=3.7',
        )
        """)

    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

        with open(os.path.join(path, '.gitkeep'), 'w') as fp:
            pass


if __name__ == "__main__":
    create_directory_structure()
