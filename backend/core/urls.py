from django.urls import path

from core.api.private_chat import PrivateMessagesView, SharedKnowledgeView
from core.views import HealthView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("projects/<uuid:project_id>/messages/", PrivateMessagesView.as_view(), name="private-messages"),
    path("projects/<uuid:project_id>/knowledge/", SharedKnowledgeView.as_view(), name="shared-knowledge"),
]
