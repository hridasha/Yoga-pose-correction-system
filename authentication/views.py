from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import views as auth_views
from .forms import RegisterForm, LoginForm, PasswordChangeForm
from .models import CustomUser
import os
import json

# def register(request):
#     if request.method == 'POST':
#         form = RegisterForm(request.POST)
#         if form.is_valid():
#             user = form.save(commit=False)
#             user.save()
#             login(request, user)
#             messages.success(request, 'Registration successful! Welcome to Yoga PC.')
#             return redirect('home')
#         else:
#             messages.error(request, 'Registration failed. Please check the errors below.')
#     else:
#         form = RegisterForm()
#     return render(request, 'authentication/register.html', {'form': form})



def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        
        if form.is_valid():
            user = form.save()

            authenticated_user = authenticate(
                request, 
                username=user.username, 
                password=request.POST['password1']
            )

            if authenticated_user:
                login(request, authenticated_user)
                messages.success(request, 'Registration successful! Welcome to Yoga PC.')
                return redirect('home')
            else:
                messages.error(request, 'Authentication failed. Please log in manually.')
                return redirect('login')

        else:
            messages.error(request, f'Registration failed: {form.errors}')
    
    else:
        form = RegisterForm()

    return render(request, 'authentication/register.html', {'form': form})


def custom_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            
            try:
                user = CustomUser.objects.get(email=email)
                if user.check_password(password):
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.get_full_name() or user.email}!')
                    return redirect('home')
                else:
                    messages.error(request, 'Invalid email or password.')
            except CustomUser.DoesNotExist:
                messages.error(request, 'Invalid email or password.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = LoginForm()
    
    return render(request, 'authentication/login.html', {
        'form': form,
        'next': request.GET.get('next', '')
    })


# @login_required
# def profile(request):
#     if request.method == 'POST':
#         user = request.user
#         user.first_name = request.POST.get('first_name', '')
#         user.last_name = request.POST.get('last_name', '')
#         user.age = request.POST.get('age', '')
#         user.save()
#         messages.success(request, 'Profile updated successfully!')
#         return redirect('profile')

    
#     return render(request, 'authentication/profile.html', {'user': request.user})

@login_required
def profile(request):
    if request.method == 'POST':
        user = request.user
        user.first_name = request.POST.get('first_name', '')
        user.last_name = request.POST.get('last_name', '')
        user.age = request.POST.get('age', '')

        selected_photo_id = request.POST.get('selected_photo')
        if selected_photo_id:
            user.profile_photo = f'images/profile{selected_photo_id}.png'  

        user.save()
        messages.success(request, 'Profile updated successfully!')
        return redirect('profile')

    return render(request, 'authentication/profile.html', {'user': request.user})



@login_required
def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('home')

@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.POST)
        if form.is_valid():
            user = request.user
            if user.check_password(form.cleaned_data['old_password']):
                user.set_password(form.cleaned_data['new_password1'])
                user.save()
                messages.success(request, 'Your password has been successfully changed.')
                return redirect('profile')
            else:
                messages.error(request, 'Old password is incorrect.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PasswordChangeForm()
    return render(request, 'authentication/change_password.html', {'form': form})

def password_reset_request(request):
    if request.method == "POST":
        email = request.POST.get('email')
        try:
            user = CustomUser.objects.get(email=email)
            # Generate token
            token = default_token_generator.make_token(user)
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            
            subject = "Password Reset Requested"
            email_template_name = "authentication/password_reset_email.html"
            c = {
                "email": user.email,
                'domain': '127.0.0.1:8000',
                'site_name': 'Yoga PC',
                "uid": uid,
                "user": user,
                'token': token,
                'protocol': 'http',
            }
            email = render_to_string(email_template_name, c)
            try:
                send_mail(subject, email, settings.EMAIL_HOST_USER, [user.email], fail_silently=False)
                messages.success(request, 'A message with reset password instructions has been sent to your inbox.')
                return redirect('password_reset_done')
            except Exception as e:
                messages.error(request, f"Error sending email: {str(e)}")
                return redirect('password_reset')
        except CustomUser.DoesNotExist:
            messages.error(request, 'No user found with this email address.')
            return redirect('password_reset')
    
    return render(request, 'authentication/password_reset.html')

def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = CustomUser.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            password1 = request.POST.get('password1')
            password2 = request.POST.get('password2')
            
            if password1 and password2 and password1 == password2:
                user.set_password(password1)
                user.save()
                messages.success(request, 'Your password has been reset successfully.')
                return redirect('login')
            else:
                messages.error(request, 'Passwords do not match.')
        
        return render(request, 'authentication/password_reset_confirm.html', {
            'validlink': True,
            'uidb64': uidb64,
            'token': token
        })
    else:
        messages.error(request, 'The reset password link is no longer valid.')
        return render(request, 'authentication/password_reset_confirm.html', {'validlink': False})

@login_required
def update_profile_photo(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            photo_id = data.get('photo_id')
            
            if photo_id:
                request.user.profile_photo = f'images/profile{photo_id}.png'
                request.user.save()
                return JsonResponse({'success': True})
            else:
                return JsonResponse({'success': False, 'error': 'No photo ID provided'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})