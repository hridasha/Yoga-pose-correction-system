from django.contrib import admin
from .models import YogaPoseIdealAngle

# Register the model
@admin.register(YogaPoseIdealAngle)
class YogaPoseIdealAngleAdmin(admin.ModelAdmin):
    list_display = ('pose_name', 'view', 'is_flipped')  # Display columns in admin
    list_filter = ('pose_name', 'view', 'is_flipped')   # Filter options
    search_fields = ('pose_name', 'view')               # Search bar fields
