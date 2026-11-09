from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from core.models import Project, ProjectMembership, User

admin.site.register(User, UserAdmin)


class MembershipInline(admin.TabularInline):
    model = ProjectMembership
    extra = 0


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("title", "id", "created_at")
    inlines = [MembershipInline]
