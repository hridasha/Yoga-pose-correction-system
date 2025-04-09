import os
from django.core.wsgi import get_wsgi_application
from django.core.files import File

# Initialize Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ypc.settings')
application = get_wsgi_application()

from pose_selection.models import YogaPoseImage

def import_pose_images():
    dataset_path = 'D:/YogaPC/ypc/datasets/pose_images'
    
    print("Starting image import...")
    
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
            pose_name = os.path.splitext(filename)[0]
            try:
                # Create a new YogaPoseImage instance
                image = YogaPoseImage(pose_name=pose_name)
                with open(os.path.join(dataset_path, filename), 'rb') as f:
                    django_file = File(f)
                    image.image.save(filename, django_file, save=True)
                print(f"Successfully imported {filename}")
            except Exception as e:
                print(f"Error importing {filename}: {str(e)}")

    print("Image import completed!")

if __name__ == '__main__':
    import_pose_images()