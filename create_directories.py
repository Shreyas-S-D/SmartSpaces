import os

def create_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'uploads',
        'static',
        'templates',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()