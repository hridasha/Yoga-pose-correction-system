from django.contrib import admin
from .models import YogaPoseIdealAngle, YogaPoseDetails, YogaPoseHold

# Register the model
@admin.register(YogaPoseIdealAngle)
class YogaPoseIdealAngleAdmin(admin.ModelAdmin):
    list_display = ('pose_name', 'view', 'is_flipped') 
    list_filter = ('pose_name', 'view', 'is_flipped')   
    search_fields = ('pose_name', 'view')              
     
@admin.register(YogaPoseDetails)
class YogaPoseDetailsAdmin(admin.ModelAdmin):
    list_display = ('pose_name', 'english_name', 'benefits', 'level', 'hold_duration')
    search_fields = ('pose_name', 'english_name', 'benefits', 'level')
    list_filter = ('pose_name', 'level')

@admin.register(YogaPoseHold)
class YogaPoseHoldAdmin(admin.ModelAdmin):
    list_display = ('pose_name', 'english_name', 'hold_duration')
    search_fields = ('pose_name', 'english_name')