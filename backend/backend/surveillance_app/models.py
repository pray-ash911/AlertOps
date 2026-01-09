from django.db import models
from django.contrib.auth.models import User

# --- Choices for Event Status ---
EVENT_STATUS_CHOICES = [
    ('NEW', 'New'),
    ('REVIEWED', 'Reviewed - Valid'),
    ('FALSE', 'False Alarm'),
    ('CLOSED', 'Closed'),
]

# ------------------------------------------------
# 1. Lookup Tables
# ------------------------------------------------

class EventType(models.Model):
    """
    Defines the type of security alert (WEAPON, OVERCROWDING).
    """
    type_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField()

    def __str__(self):
        return self.name


class SurveillanceArea(models.Model):
    """
    Defines a monitored physical zone (used for OVERCROWDING).
    """
    area_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    overcrowding_threshold = models.IntegerField(default=5)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class User(models.Model):
    username = models.CharField(max_length=150)
    email = models.EmailField()
    password = models.CharField(max_length=128)  # Hashed!
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

# ------------------------------------------------
# 2. Fact and Evidence Tables
# ------------------------------------------------

class EventLog(models.Model):
    """
    Logs every WEAPON or OVERCROWDING incident.
    """
    log_id = models.AutoField(primary_key=True)

    # Only these two types remain
    type = models.ForeignKey(EventType, on_delete=models.PROTECT)

    # Keep area for OVERCROWDING
    area = models.ForeignKey(
        SurveillanceArea,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Area where the overcrowding occurred."
    )

    timestamp = models.DateTimeField(auto_now_add=True)

    # Weapon → confidence
    # Overcrowding → people count
    confidence_value = models.FloatField()

    status = models.CharField(
        max_length=10,
        choices=EVENT_STATUS_CHOICES,
        default='NEW'
    )

    # Link to admin user who reviewed/closed the event
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_events',
        help_text="Admin user who reviewed or handled this event"
    )
    
    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the event was reviewed/closed by admin"
    )

    def __str__(self):
        return f"{self.type.name} Alert at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"


class EventEvidence(models.Model):
    """
    Stores snapshot images for an event.
    """
    evidence_id = models.AutoField(primary_key=True)
    log = models.ForeignKey(EventLog, on_delete=models.CASCADE, related_name='evidence')
    file_path = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50, default='image/jpeg')

    def __str__(self):
        return f"Evidence for Log ID {self.log.log_id}"
