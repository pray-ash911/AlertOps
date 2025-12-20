import cv2
import os
import time
import traceback
import requests
import urllib.parse
import numpy as np
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from datetime import timedelta
from django.utils import timezone
from django.utils.timezone import localtime
from django.conf import settings
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
# --- NEW: Import the new database models ---
from .models import EventLog, EventType, EventEvidence, SurveillanceArea

# --- REQUIRED IMPORTS AND MODEL INITIALIZATION ---

# Initialize global variables (needed even if models fail to load)
WEAPON_MODEL = None
CROWD_MODEL = None
WEAPON_EVENT_TYPE_OBJ = None
WEAPON_EVENT_TYPE_ID = None
OVERCROWDING_EVENT_TYPE_OBJ = None
OVERCROWDING_EVENT_TYPE_ID = None

try:
    from ultralytics import YOLO

    print("Ultralytics YOLO imported successfully.")

   # In your views.py, replace the model loading section:

    # --- MODEL 1: WEAPON DETECTION ---
    MODEL_FILE_NAME_WEAPON = 'best (1).pt'
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH_WEAPON = os.path.join(PROJECT_ROOT, 'models', MODEL_FILE_NAME_WEAPON)

    import torch

    # Check CUDA BEFORE loading models
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

    if not os.path.exists(MODEL_PATH_WEAPON):
        print(f"ERROR: Model file not found at: {MODEL_PATH_WEAPON}")
        WEAPON_MODEL = None
    else:
        try:
            # Load model and immediately move to device
            WEAPON_MODEL = YOLO(MODEL_PATH_WEAPON)
            WEAPON_MODEL.to(device)  # Force to GPU or CPU
            print(f"YOLO (Weapon Detection) Model loaded successfully on: {WEAPON_MODEL.device}")
            print(f"Weapon Model Names: {WEAPON_MODEL.names}")
        except Exception as e:
            print(f"ERROR loading YOLO model: {e}")
            WEAPON_MODEL = None

    # --- MODEL 2: OVERCROWDING DETECTION ---
    MODEL_FILE_NAME_CROWD = 'yolov8m.pt'
    MODEL_PATH_CROWD = os.path.join(PROJECT_ROOT, 'models', MODEL_FILE_NAME_CROWD)

    try:
        # Load the model and move to device
        CROWD_MODEL = YOLO(MODEL_PATH_CROWD)
        CROWD_MODEL.to(device)  # Force to same device
        print(f"YOLO (Overcrowding Detection) Model loaded successfully on: {CROWD_MODEL.device}")
        print(f"Crowd Model has {len(CROWD_MODEL.names)} classes")
    except Exception as e:
        print(f"ERROR loading YOLO crowd model: {e}")
        CROWD_MODEL = None


    # Add verification after moving to GPU
    import torch
    if torch.cuda.is_available():
        WEAPON_MODEL.to('cuda:0')
        # Verify
        print(f"Weapon Model device: {WEAPON_MODEL.device}")
        print(f"Is model on GPU?: {next(WEAPON_MODEL.model.parameters()).device}")

   # --- CONFIGURATION FOR WEAPON DETECTION ---
    INFERENCE_SKIP_FRAMES = 0  # Process EVERY frame for maximum detection
    FIREARM_KEYWORDS = ['gun', 'pistol', 'handgun', 'rifle', 'firearm', 'weapon']
    BLADE_KEYWORDS = ['knife', 'sword', 'blade', 'dagger']
    WEAPON_KEYWORDS = FIREARM_KEYWORDS + BLADE_KEYWORDS

    # CRITICAL: LOWER THESE VALUES FOR REAL-WORLD DETECTION
    WEAPON_LOG_CONFIDENCE = 0.25  # Confidence for logging to database
    WEAPON_DETECTION_CONFIDENCE = 0.20  # Confidence for real-time display
    LOG_COOLDOWN_SECONDS = 3

    # --- CONFIGURATION FOR OVERCROWDING DETECTION ---
    OVERCROWDING_THRESHOLD = 3  # Start with very low threshold
    CROWD_CONFIDENCE_THRESHOLD = 0.25  # People need lower confidence
    CROWD_LOG_COOLDOWN_SECONDS = 15

    # In your detection loop, add debugging:
    print(f"Weapon Model Names: {list(WEAPON_MODEL.names.values())}")
    print(f"Crowd Model Names: {list(CROWD_MODEL.names.values())}")

except Exception as e:
    WEAPON_MODEL = None
    CROWD_MODEL = None
    WEAPON_EVENT_TYPE_OBJ = None
    WEAPON_EVENT_TYPE_ID = None
    OVERCROWDING_EVENT_TYPE_OBJ = None
    OVERCROWDING_EVENT_TYPE_ID = None
    print(f"CRITICAL ERROR initializing YOLO models: {e}")
    traceback.print_exc()


# --- NEW: Helper function to get or create the 'WEAPON' EventType ---
def get_or_create_weapon_event_type():
    """
    Helper function to retrieve the 'WEAPON' EventType from the database,
    creating it if it doesn't exist. This function should be called only
    when needed, after Django has initialized the database connection.
    """
    global WEAPON_EVENT_TYPE_OBJ, WEAPON_EVENT_TYPE_ID # Access the global variables
    try:
        if WEAPON_EVENT_TYPE_OBJ is None: # Only fetch/create if not already done
            weapon_event_type, created = EventType.objects.get_or_create(
                name='WEAPON',
                defaults={'description': 'A weapon (e.g., gun, knife) has been detected.'}
            )
            WEAPON_EVENT_TYPE_OBJ = weapon_event_type # Store the object
            WEAPON_EVENT_TYPE_ID = weapon_event_type.type_id # Store the ID
            if created:
                print(f"Created new EventType: {weapon_event_type.name} (ID: {WEAPON_EVENT_TYPE_ID})")
            else:
                print(f"Found existing EventType: {weapon_event_type.name} (ID: {WEAPON_EVENT_TYPE_ID})")
        return WEAPON_EVENT_TYPE_OBJ
    except Exception as e:
        print(f"ERROR in get_or_create_weapon_event_type: {e}")
        traceback.print_exc()
        return None


# --- NEW: Helper function to get or create the 'OVERCROWDING' EventType ---
def get_or_create_overcrowding_event_type():
    """
    Helper function to retrieve the 'OVERCROWDING' EventType from the database,
    creating it if it doesn't exist. This function should be called only
    when needed, after Django has initialized the database connection.
    """
    global OVERCROWDING_EVENT_TYPE_OBJ, OVERCROWDING_EVENT_TYPE_ID # Access the global variables
    try:
        if OVERCROWDING_EVENT_TYPE_OBJ is None: # Only fetch/create if not already done
            overcrowding_event_type, created = EventType.objects.get_or_create(
                name='OVERCROWDING',
                defaults={'description': 'The number of people in the area exceeds the defined threshold.'}
            )
            OVERCROWDING_EVENT_TYPE_OBJ = overcrowding_event_type # Store the object
            OVERCROWDING_EVENT_TYPE_ID = overcrowding_event_type.type_id # Store the ID
            if created:
                print(f"Created new EventType: {overcrowding_event_type.name} (ID: {OVERCROWDING_EVENT_TYPE_ID})")
            else:
                print(f"Found existing EventType: {overcrowding_event_type.name} (ID: {OVERCROWDING_EVENT_TYPE_ID})")
        return OVERCROWDING_EVENT_TYPE_OBJ
    except Exception as e:
        print(f"ERROR in get_or_create_overcrowding_event_type: {e}")
        traceback.print_exc()
        return None


# --- Utility Function for Home Route ---
def home(request):
    """ Simple Django view for health check. """
    return HttpResponse("<h1>AI Surveillance System Backend is Running (Weapon & Overcrowding Detection Core)!</h1>")


# ------------------------------------------------------------------
# 1. GOOGLE FORMS Configuration and Utility (NEW - Replaces IFTTT)
# ------------------------------------------------------------------

# CORRECTED SUBMISSION URL
# We replace '/viewform?usp=header' with '/formResponse'
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdoEM-r0P7VNPcxbDiXDsvb87s0bX_xTX6_Tuw9XYh2g2YB2w/formResponse      "

# CORRECTED ENTRY IDs
FIELD_ID_IMAGE = "entry.194280707"  # Corresponds to Value1_Image_URL
FIELD_ID_LABEL = "entry.1168214022"  # Corresponds to Value2_Label_Confidence
FIELD_ID_TIME = "entry.1784554534"  # Corresponds to Value3_Timestamp

# URLs for different access levels
# NGROK_URL: Public URL via ngrok for Google Forms and API snapshot URLs
NGROK_URL = "https://883b2c5fbccc.ngrok-free.app  "
# LOCAL_URL: Local Django server URL for dashboard API calls
LOCAL_URL = "http://127.0.0.1:8000"


def send_google_form_alert(snapshot_relative_path, label, confidence):
    """
    Sends a POST request to Google Forms, triggering a submission
    and thereby the form's internal email notification.
    """
    # Check if the Ngrok URL has been updated from the placeholder
    if NGROK_URL.endswith("YOUR-COPIED-NGROK-URL-HERE"):
        print("WARNING: NGROK_URL not updated. Cannot send complete alert.")
        return

    # 1. Construct the public URL for the image.
    safe_path = urllib.parse.quote(snapshot_relative_path)
    # Example: https://883b2c5fbccc.ngrok-free.app/media/snapshots/file.jpg
    image_url = f"{NGROK_URL}{settings.MEDIA_URL}{safe_path}"

    # 2. Prepare payload using the specific Google Form entry IDs
    payload = {
        FIELD_ID_IMAGE: image_url,
        FIELD_ID_LABEL: f"{label} (Conf: {confidence:.2f})",
        FIELD_ID_TIME: localtime(timezone.now()).strftime('%Y-%m-%d %H:%M:%S %Z'),
    }

    try:
        # Send data to the form response endpoint
        response = requests.post(GOOGLE_FORM_URL, data=payload, timeout=7)

        # Google Forms usually returns 200 or 302/303 redirect if successful
        if response.status_code in [200, 302, 303]:
            print(f"Google Forms Alert sent successfully for {label}. Status: {response.status_code}")
        else:
            print(f"Google Forms Alert failed (HTTP {response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Google Forms Alert failed due to network error: {e}")


# ------------------------------------------------------------------
# 2. SINGLE FRAME DETECTION VIEW (Placeholder)
# ------------------------------------------------------------------
@api_view(['POST'])
@permission_classes([AllowAny])
def violence_detection_view(request):
    """
    DUMMY VIEW: Reintroduced to satisfy existing URL patterns in urls.py.
    This functionality is now handled by the streaming view.
    """
    return JsonResponse({'status': 'info',
                         'message': 'This endpoint is no longer active. Use /api/video-feed/ for real-time streaming.'},
                        status=200)


# ------------------------------------------------------------------
# 3. REAL-TIME STREAMING FUNCTIONS (Weapon & Overcrowding Detection Core)
# ------------------------------------------------------------------
def generate_frames():
    """
    Python generator function that continuously captures and processes
    frames using the WEAPON_MODEL for real-time weapon detection
    and the CROWD_MODEL for real-time overcrowding detection.

    Includes frame skipping for performance.
    """
    # NOTE: (The implementation of generate_frames remains the same, but the
    # call to send_ifttt_alert is replaced by send_google_form_alert below.)

    if WEAPON_MODEL is None or CROWD_MODEL is None: # Check if both models are available first
        print("Model(s) not available. Exiting stream.")
        # Yield an error frame instead of returning
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Models not loaded", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        error_frame = cv2.imencode('.jpg', error_img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
        return

    # --- GET THE EVENT TYPE OBJECTS ---
    weapon_event_type_obj = get_or_create_weapon_event_type()
    overcrowding_event_type_obj = get_or_create_overcrowding_event_type()
    if weapon_event_type_obj is None or overcrowding_event_type_obj is None:
        print("ERROR: Could not get/create EventTypes. Exiting stream.")
        # Yield an error frame
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "EventTypes not available", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        error_frame = cv2.imencode('.jpg', error_img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
        return # Exit the stream if EventType is unavailable

    # 💡 FIX: Explicitly use DirectShow backend for stability on Windows
    # (CAP_DSHOW = 700, the preferred value for reliability on Windows/MSMF issues)
    # Try multiple camera indices to find an available camera
    camera = None
    for camera_index in range(3):  # Try indices 0, 1, 2
        test_camera = None
        try:
            test_camera = cv2.VideoCapture(camera_index + cv2.CAP_DSHOW)
            if test_camera.isOpened():
                # Test if we can actually read a frame
                ret, _ = test_camera.read()
                if ret:
                    camera = test_camera
                    print(f"Camera opened successfully at index {camera_index}")
                    break
                else:
                    test_camera.release()
                    test_camera = None
        except Exception as e:
            print(f"Error trying camera index {camera_index}: {e}")
            if test_camera is not None:
                test_camera.release()

    if camera is None or not camera.isOpened():
        print("CRITICAL: No camera available at any index (0, 1, or 2). Stopping stream.")
        # Yield an error frame showing camera unavailable
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Unavailable", (150, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_img, "Please check camera connection", (100, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(error_img, "Tried indices: 0, 1, 2", (100, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        error_frame = cv2.imencode('.jpg', error_img)[1].tobytes()
        while True:  # Keep yielding error frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(1)  # Yield error frame every second
        return

    last_annotated_frame = None  # Stores the last frame annotated by the model
    frame_count = 0
    TARGET_FRAME_TIME_MS = 100  # Target 10 FPS

    last_weapon_alert_time = 0
    last_crowd_alert_time = 0 # Track last overcrowding alert time

    SNAPSHOT_DIR_NAME = ''  # No subfolder, save directly in MEDIA_ROOT

    while True:
        start_time = time.time()
        current_time = time.time()
        success, frame = camera.read()

        if not success:
            print("Could not read frame from camera. Stopping stream.")
            break

        current_frame_for_display = frame.copy()
        alert_triggered = False # Track if any alert was just logged in this cycle
        person_count = 0 # Initialize person count for this frame

        frame_count += 1
        # Determine if we run the expensive inference step
        run_inference = (frame_count % (INFERENCE_SKIP_FRAMES + 1)) == 0

        # --- Detection and Annotation Logic ---
        try:

            if run_inference:
                # --- RUN EXPENSIVE YOLO INFERENCE FOR WEAPON ---
                weapon_results = WEAPON_MODEL.predict(frame, verbose=False, conf=WEAPON_LOG_CONFIDENCE)

                # --- RUN EXPENSIVE YOLO INFERENCE FOR CROWD (People) ---
                crowd_results = CROWD_MODEL.predict(frame, verbose=False, conf=0.25) # Lower confidence for people detection

                # Use the model's built-in plotting to create the annotated frame
                # Combine the annotations from both models if possible, or use one.
                # For simplicity, let's use the weapon model's annotation if it has detections, otherwise use crowd model's.
                if weapon_results and weapon_results[0] and len(weapon_results[0].boxes) > 0:
                    last_annotated_frame = weapon_results[0].plot()
                elif crowd_results and crowd_results[0] and len(crowd_results[0].boxes) > 0:
                    last_annotated_frame = crowd_results[0].plot()
                else:
                    # If no results from either model, use the raw frame as the last annotated frame
                    last_annotated_frame = current_frame_for_display.copy()

            # --- APPLY ANNOTATIONS (Use the last known annotated frame or the current raw frame) ---
            if last_annotated_frame is not None:
                current_frame_for_display = last_annotated_frame.copy()

            # If any results were returned (for logging checks) AND we ran inference
            if run_inference and ((weapon_results and weapon_results[0]) or (crowd_results and crowd_results[0])):

                # --- PROCESS WEAPON RESULTS ---
                if weapon_results and weapon_results[0].boxes:
                    weapon_confs = weapon_results[0].boxes.conf.cpu().numpy()
                    weapon_classes = weapon_results[0].boxes.cls.int().cpu().tolist()
                    weapon_names = WEAPON_MODEL.names

                    # Check for any weapon detection
                    for conf, cls in zip(weapon_confs, weapon_classes):
                        class_name = weapon_names.get(cls, "unknown").lower()

                        if any(keyword in class_name for keyword in WEAPON_KEYWORDS):

                            # --- STANDARDIZE LABEL FOR LOGGING ---
                            final_weapon_type = "UNKNOWN_WEAPON"
                            if any(keyword in class_name for keyword in FIREARM_KEYWORDS):
                                final_weapon_type = "FIREARM"
                            elif any(keyword in class_name for keyword in BLADE_KEYWORDS):
                                final_weapon_type = "BLADE"

                            label = f"WEAPON_{final_weapon_type}"

                            # --- LOGGING CHECK: WEAPON DETECTED (Cooldown enforced) ---
                            # Only log if no other alert (like overcrowding) was just logged in this cycle
                            if current_time - last_weapon_alert_time > LOG_COOLDOWN_SECONDS and not alert_triggered:
                                # --- FILE PATH HANDLING (Using MEDIA_ROOT logic for persistence) ---
                                SNAPSHOT_FULL_PATH = settings.MEDIA_ROOT  # Save directly in MEDIA_ROOT
                                os.makedirs(SNAPSHOT_FULL_PATH, exist_ok=True)

                                filename = f"{label}_{int(current_time)}.jpg"
                                filepath = os.path.join(SNAPSHOT_FULL_PATH, filename)  # Full path on the server
                                db_snapshot_path = filename  # Store only the filename, MEDIA_URL handles the path

                                # Log the snapshot of the annotated frame
                                cv2.imwrite(filepath, current_frame_for_display)

                                # --- NEW: Log to EventLog and EventEvidence models for WEAPON ---
                                # 1. Create the main EventLog entry
                                event_log = EventLog.objects.create(
                                    type=weapon_event_type_obj,  # Use the EventType object obtained via the helper function
                                    # area=None, # Optional: Link to a SurveillanceArea if applicable
                                    timestamp=timezone.now(),
                                    confidence_value=conf,  # Store the confidence score
                                    # status defaults to 'NEW'
                                )

                                # 2. Create the EventEvidence entry linked to the EventLog
                                EventEvidence.objects.create(
                                    log=event_log,  # Link to the EventLog just created
                                    file_path=db_snapshot_path,  # Store the relative path
                                    file_type='image/jpeg'  # Specify the file type
                                )

                                print(
                                    f"Logged {label} event (Log ID: {event_log.log_id}): {db_snapshot_path}, Confidence: {conf:.2f}")

                                # --- GOOGLE FORMS ALERT TRIGGER (NEW) for WEAPON ---
                                send_google_form_alert(db_snapshot_path, label, conf)

                                last_weapon_alert_time = current_time
                                alert_triggered = True # Prevent overcrowding alert from triggering in this cycle if one just happened

                            # Overlay primary system alert message for weapon
                            cv2.putText(current_frame_for_display, f"!!! {label} DETECTED !!!", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

                # --- PROCESS CROWD RESULTS (People Counting) ---
                if crowd_results and crowd_results[0].boxes:
                    crowd_confs = crowd_results[0].boxes.conf.cpu().numpy()
                    crowd_classes = crowd_results[0].boxes.cls.int().cpu().tolist()
                    crowd_names = CROWD_MODEL.names

                    # --- COUNT PEOPLE FOR OVERCROWDING ---
                    for cls in crowd_classes:
                        class_name = crowd_names.get(cls, "unknown").lower()
                        if class_name == 'person': # Count all detected persons
                            person_count += 1

                    # --- CHECK FOR OVERCROWDING ALERT (Lower Priority) ---
                    # Only check/log crowd if no weapon alert was just logged in this cycle
                    if person_count > OVERCROWDING_THRESHOLD and not alert_triggered:
                        if current_time - last_crowd_alert_time > CROWD_LOG_COOLDOWN_SECONDS:
                            crowd_label = 'OVERCROWDING'

                            # --- FILE PATH HANDLING (Using MEDIA_ROOT logic for persistence) ---
                            SNAPSHOT_FULL_PATH = settings.MEDIA_ROOT  # Save directly in MEDIA_ROOT
                            os.makedirs(SNAPSHOT_FULL_PATH, exist_ok=True)

                            filename = f"{crowd_label}{person_count}{int(current_time)}.jpg" # Include count in filename
                            filepath = os.path.join(SNAPSHOT_FULL_PATH, filename)  # Full path on the server
                            db_snapshot_path = filename  # Store only the filename, MEDIA_URL handles the path

                            # Log the snapshot of the annotated frame showing the crowd
                            cv2.imwrite(filepath, current_frame_for_display)

                            # --- NEW: Log to EventLog and EventEvidence models for OVERCROWDING ---
                            # 1. Create the main EventLog entry for OVERCROWDING
                            # NOTE: confidence_value stores the COUNT for overcrowding
                            event_log = EventLog.objects.create(
                                type=overcrowding_event_type_obj,  # Use the OVERCROWDING EventType object
                                # area=None, # Optional: Link to a SurveillanceArea if applicable later
                                timestamp=timezone.now(),
                                confidence_value=person_count,  # Store the count
                                # status defaults to 'NEW'
                            )

                            # 2. Create the EventEvidence entry linked to the EventLog
                            EventEvidence.objects.create(
                                log=event_log,  # Link to the EventLog just created
                                file_path=db_snapshot_path,  # Store the relative path
                                file_type='image/jpeg'  # Specify the file type
                            )

                            print(
                                f"Logged {crowd_label} event (Log ID: {event_log.log_id}): {db_snapshot_path}, Count: {person_count}")

                            # --- GOOGLE FORMS ALERT TRIGGER (NEW) for OVERCROWDING ---
                            send_google_form_alert(db_snapshot_path, crowd_label, person_count)

                            last_crowd_alert_time = current_time
                            alert_triggered = True # Prevent weapon alert from triggering in this cycle if one just happened

                        # Overlay primary system alert message for overcrowding
                        cv2.putText(current_frame_for_display, f"!!! OVERCROWDING: {person_count}/{OVERCROWDING_THRESHOLD} !!!", (10, 100), # Changed position
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3) # Different color/font size


            # --- Display Current Person Count (Always) ---
            # Show the current count on the video feed, regardless of alerts
            cv2.putText(current_frame_for_display,
                        f"People: {person_count} (Threshold: {OVERCROWDING_THRESHOLD})",
                        (10, current_frame_for_display.shape[0] - 20), # Position at bottom
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White text


            # Annotate the display frame if a non-logging alert was recently triggered
            # (This might be less relevant now with two types, but keep for weapon if needed)
            if not alert_triggered and 'last_weapon_alert_time' in locals() and current_time - last_weapon_alert_time < LOG_COOLDOWN_SECONDS:
                cv2.putText(current_frame_for_display, f"!!! RECENT WEAPON ALERT !!!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4)

        except Exception as e:
            print(f"YOLO Frame processing error: {e}")
            traceback.print_exc()

            # Annotate the display frame to show the error
            cv2.putText(current_frame_for_display, "MODEL ERROR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),
                        4)

        # --- Encode the processed frame to JPEG ---
        ret, buffer = cv2.imencode('.jpg', current_frame_for_display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # Frame Rate Control
        elapsed_time_ms = (time.time() - start_time) * 1000
        wait_time_ms = max(1, int(TARGET_FRAME_TIME_MS - elapsed_time_ms))
        time.sleep(wait_time_ms / 1000)

    # Cleanup
    camera.release()


@api_view(['GET'])
@permission_classes([AllowAny])
def video_feed_view(request):
    """
    This view returns a StreamingHttpResponse that streams the weapon/crowd detection feed.
    """
    if WEAPON_MODEL is None or CROWD_MODEL is None:  # Check if both models are loaded
        return HttpResponse('Detection Models failed to load.', status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ------------------------------------------------------------------
# 4. EVENT LOGS VIEW & STATUS (Updated for New Schema)
# ------------------------------------------------------------------

# --- NEW: Updated Event Logs View using EventLog and EventEvidence ---
@api_view(['GET'])
@permission_classes([AllowAny])
def event_logs_view(request):
    """ Returns a JSON list of all logged events for the dashboard. """
    try:
        # Query EventLog entries, joining with EventType and prefetching EventEvidence
        # Order by timestamp descending, limit to last 100
        # Use prefetch_related for the reverse foreign key (EventLog -> EventEvidence)
        events = EventLog.objects.select_related('type', 'area').prefetch_related('evidence').all().order_by('-timestamp')[:100]

        data = []
        for event in events:
            local_timestamp = localtime(event.timestamp)

            # Get the first piece of evidence (snapshot) for this event
            # Use the prefetched evidence manager
            evidence = event.evidence.first()
            snapshot_url = None
            snapshot_path = None
            if evidence:
                snapshot_path = evidence.file_path # Get the relative path from EventEvidence
                # --- Generate public snapshot URL for dashboard (NEW) ---
                if NGROK_URL != "https://YOUR-COPIED-NGROK-URL-HERE" and NGROK_URL != "YOUR_NGROK_OR_PUBLIC_URL_HERE":
                    safe_path = urllib.parse.quote(snapshot_path)
                    snapshot_url = f"{NGROK_URL}{settings.MEDIA_URL}{safe_path}"

            data.append({
                'id': event.log_id,  # Use the primary key from EventLog
                'timestamp': local_timestamp.isoformat(),
                'label': event.type.name,  # Get the name from the related EventType
                'confidence': event.confidence_value,  # Use confidence_value from EventLog (Conf for WEAPON, Count for CROWD)
                'snapshot_url': snapshot_url,
                'snapshot_path': snapshot_path,  # Include the path from EventEvidence
            })

        return JsonResponse(data, safe=False, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error in event_logs_view: {e}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- NEW: Updated Latest Status View using EventLog (Handles Weapon & Overcrowding) ---
# --- NEW: Updated Latest Status View using EventLog (Handles Weapon & Overcrowding) ---
@api_view(['GET'])
@permission_classes([AllowAny])
def get_latest_status(request):
    """API endpoint for Streamlit to poll for the most recent alert status."""
    try:
        # Query the latest EventLog entry
        latest_event = EventLog.objects.select_related('type').latest('timestamp')  # Join with EventType
        alert_window = timedelta(seconds=30)
        is_recent_alert = (timezone.now() - latest_event.timestamp) < alert_window

        # Check if the latest event is a 'WEAPON' or 'OVERCROWDING' event
        event_type_name = latest_event.type.name.upper()
        is_monitored_alert = event_type_name in ['WEAPON', 'OVERCROWDING']

        local_timestamp = localtime(latest_event.timestamp)

        if is_recent_alert and is_monitored_alert:
            # Customize message based on event type
            if event_type_name == 'WEAPON':
                message_details = f"Conf: {latest_event.confidence_value:.2f}"
            elif event_type_name == 'OVERCROWDING':
                # For overcrowding, confidence_value holds the count
                message_details = f"Count: {int(latest_event.confidence_value)}"

            status_data = {
                'status_level': 'ALERT',  # ✅ FIXED - consistent single quotes
                'message': f"!!! {event_type_name} DETECTED at {local_timestamp.strftime('%H:%M:%S')} ({message_details}) !!!",
                'confidence': latest_event.confidence_value
            }
        else:
            status_data = {
                'status_level': 'OK',
                'message': 'System operational. Monitoring live stream.',
                'confidence': 0.0
            }

    # Catch the DoesNotExist exception from EventLog
    except EventLog.DoesNotExist:
        status_data = {
            'status_level': 'IDLE',
            'message': 'System operational. Waiting for first event log.',
            'confidence': 0.0
        }
    except Exception as e:
        print(f"Error in get_latest_status: {e}")
        traceback.print_exc()
        status_data = {
            'status_level': 'ERROR',
            'message': f'System Error: {str(e)}',
            'confidence': 0.0
        }

    return JsonResponse(status_data, safe=False)


# --- NEW: Analytics View for Monthly Trends ---
@api_view(['GET'])
@permission_classes([AllowAny])
def analytics_view(request):
    """Returns JSON data for monthly trends (last 30 days) of events grouped by date and type."""
    try:
        # Calculate the date 30 days ago
        thirty_days_ago = timezone.now() - timedelta(days=30)
        
        # Query EventLog for the last 30 days, filter by WEAPON and OVERCROWDING types
        events = EventLog.objects.filter(
            timestamp__gte=thirty_days_ago,
            type__name__in=['WEAPON', 'OVERCROWDING']
        ).annotate(
            date=TruncDate('timestamp')
        ).values('date', 'type__name').annotate(
            count=Count('log_id')
        ).order_by('date', 'type__name')
        
        # Prepare data for JSON response
        data = {}
        for event in events:
            date_str = event['date'].strftime('%Y-%m-%d')  # Format as date only
            event_type = event['type__name']
            count = event['count']
            
            if date_str not in data:
                data[date_str] = {'date': date_str, 'weapon': 0, 'overcrowding': 0, 'total_detections': 0}

            if event_type == 'WEAPON':
                data[date_str]['weapon'] = count
                data[date_str]['total_detections'] += count
            elif event_type == 'OVERCROWDING':
                data[date_str]['overcrowding'] = count
                data[date_str]['total_detections'] += count
        
        # Convert to list and sort by date
        result = sorted(data.values(), key=lambda x: x['date'])
        
        # Get hourly data (last 7 days)
        seven_days_ago = timezone.now() - timedelta(days=7)
        
        # Get weapon events for hourly data
        weapon_hourly = EventLog.objects.filter(
            timestamp__gte=seven_days_ago,
            type__name='WEAPON'
        ).extra({
            'hour': "EXTRACT(HOUR FROM timestamp)"
        }).values('hour').annotate(
            count=Count('log_id')
        ).order_by('hour')
        
        # Get overcrowding events for hourly data
        crowd_hourly = EventLog.objects.filter(
            timestamp__gte=seven_days_ago,
            type__name='OVERCROWDING'
        ).extra({
            'hour': "EXTRACT(HOUR FROM timestamp)"
        }).values('hour').annotate(
            count=Count('log_id')
        ).order_by('hour')
        
        # Prepare hourly data structure
        hourly_data = []
        for hour in range(24):
            weapon_count = next((item['count'] for item in weapon_hourly if int(item['hour']) == hour), 0)
            crowd_count = next((item['count'] for item in crowd_hourly if int(item['hour']) == hour), 0)
            
            hourly_data.append({
                'hour': hour,
                'weapon': weapon_count,
                'overcrowding': crowd_count,
                'total': weapon_count + crowd_count
            })
        
        # Calculate summary statistics
        total_weapons = sum(item['weapon'] for item in result)
        total_overcrowding = sum(item['overcrowding'] for item in result)
        
        # Find peak hour
        if hourly_data:
            peak_hour_item = max(hourly_data, key=lambda x: x['total'])
            peak_hour = peak_hour_item['hour']
            peak_weapon = peak_hour_item['weapon']
            peak_crowd = peak_hour_item['overcrowding']
        else:
            peak_hour = 14  # Default peak hour
            peak_weapon = 0
            peak_crowd = 0
        
        # Get recent events
        recent_events = EventLog.objects.filter(
            timestamp__gte=thirty_days_ago
        ).select_related('type').order_by('-timestamp')[:10].values(
            'log_id',
            'timestamp',
            'type__name',
            'confidence_value',
            'status'
        )
        
        # Convert timestamps to string format
        recent_events_list = []
        for event in recent_events:
            event_dict = dict(event)
            event_dict['timestamp'] = event_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            recent_events_list.append(event_dict)
        
        # Get today's counts
        today = timezone.now().date()
        today_weapon = EventLog.objects.filter(
            timestamp__date=today,
            type__name='WEAPON'
        ).count()
        
        today_crowd = EventLog.objects.filter(
            timestamp__date=today,
            type__name='OVERCROWDING'
        ).count()
        
        # Get event type distribution
        type_distribution = EventLog.objects.filter(
            timestamp__gte=thirty_days_ago
        ).values('type__name').annotate(
            count=Count('log_id')
        )
        
        response_data = {
            'daily_analytics': result,
            'hourly_analytics': hourly_data,
            'recent_events': recent_events_list,
            'type_distribution': list(type_distribution),
            'summary': {
                'total_weapons': total_weapons,
                'total_overcrowding': total_overcrowding,
                'total_all': total_weapons + total_overcrowding,
                'date_range': {
                    'start': thirty_days_ago.strftime('%Y-%m-%d'),
                    'end': timezone.now().strftime('%Y-%m-%d')
                },
                'peak_hour': f"{int(peak_hour):02d}:00",
                'peak_hour_weapon': peak_weapon,
                'peak_hour_crowd': peak_crowd,
                'avg_daily_weapons': round(total_weapons / max(len(result), 1), 1),
                'avg_daily_crowd': round(total_overcrowding / max(len(result), 1), 1),
                'today_weapon': today_weapon,
                'today_crowd': today_crowd
            }
        }
        
        return JsonResponse(response_data, safe=False, status=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"Error in analytics_view: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return sample data for development
        return JsonResponse({
            'daily_analytics': generate_sample_daily_data(),
            'hourly_analytics': generate_sample_hourly_data(),
            'recent_events': generate_sample_recent_events(),
            'type_distribution': [
                {'type__name': 'WEAPON', 'count': 45},
                {'type__name': 'OVERCROWDING', 'count': 120}
            ],
            'summary': {
                'total_weapons': 45,
                'total_overcrowding': 120,
                'total_all': 165,
                'date_range': {
                    'start': (timezone.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'end': timezone.now().strftime('%Y-%m-%d')
                },
                'peak_hour': "14:00",
                'peak_hour_weapon': 8,
                'peak_hour_crowd': 15,
                'avg_daily_weapons': 1.5,
                'avg_daily_crowd': 4.0,
                'today_weapon': 2,
                'today_crowd': 8
            }
        }, status=status.HTTP_200_OK)


def generate_sample_daily_data():
    """Generate sample daily analytics data for testing"""
    import random
    from datetime import datetime, timedelta
    
    analytics_data = []
    end_date = datetime.now().date()
    
    for i in range(30):
        date = end_date - timedelta(days=30-i-1)
        
        # Generate realistic data with patterns
        # Weekends have more events
        if date.weekday() in [4, 5]:  # Friday, Saturday
            weapon = random.randint(0, 5)
            crowd = random.randint(5, 15)
        # Sundays moderate
        elif date.weekday() == 6:  # Sunday
            weapon = random.randint(0, 3)
            crowd = random.randint(3, 10)
        # Weekdays
        else:
            weapon = random.randint(0, 3)
            crowd = random.randint(2, 8)
        
        # Add some trend
        if i > 20:  # Last 10 days
            weapon += random.randint(0, 2)
            crowd += random.randint(0, 5)
        
        analytics_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'weapon': weapon,
            'overcrowding': crowd,
            'total_detections': weapon + crowd
        })
    
    return analytics_data


def generate_sample_hourly_data():
    """Generate sample hourly data for testing"""
    import random
    
    hourly_data = []
    
    for hour in range(24):
        # Peak hours 9 AM to 9 PM
        if 9 <= hour <= 21:
            weapon = random.randint(0, 3)
            crowd = random.randint(2, 10)
        # Off-peak hours
        else:
            weapon = random.randint(0, 1)
            crowd = random.randint(0, 3)
        
        hourly_data.append({
            'hour': hour,
            'weapon': weapon,
            'overcrowding': crowd,
            'total': weapon + crowd
        })
    
    return hourly_data


def generate_sample_recent_events():
    """Generate sample recent events for testing"""
    import random
    from datetime import datetime, timedelta
    
    recent_events = []
    event_types = ['WEAPON', 'OVERCROWDING']
    statuses = ['NEW', 'REVIEWED', 'CLOSED']
    
    for i in range(10):
        event_time = datetime.now() - timedelta(hours=random.randint(0, 72))
        event_type = random.choice(event_types)
        
        if event_type == 'WEAPON':
            confidence = round(random.uniform(0.65, 0.95), 2)
        else:
            confidence = random.randint(5, 25)  # Count for overcrowding
        
        recent_events.append({
            'log_id': 1000 + i,
            'timestamp': event_time.strftime('%Y-%m-%d %H:%M:%S'),
            'type__name': event_type,
            'confidence_value': confidence,
            'status': random.choice(statuses)
        })
    
    return recent_events


# ------------------------------------------------------------------
# 5. AUTHENTICATION VIEWS (Login & Register for Admin Users)
# ------------------------------------------------------------------

from django.contrib.auth import login, logout
from .serializers import UserRegistrationSerializer, UserLoginSerializer, UserSerializer


@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def register_view(request):
    """
    Register a new admin user.
    
    GET /api/register/ - Redirects to registration page
    POST /api/register/
    Body: {
        "username": "admin",
        "email": "admin@example.com",
        "password": "password123",
        "password_confirm": "password123",
        "first_name": "Admin",
        "last_name": "User"
    }
    """
    if request.method == 'GET':
        from django.shortcuts import redirect
        return redirect('register_page')
    
    serializer = UserRegistrationSerializer(data=request.data)
    
    if serializer.is_valid():
        user = serializer.save()
        # Don't auto-login, redirect to login page
        # Return user info (excluding password)
        user_data = UserSerializer(user).data
        return Response({
            'message': 'User registered successfully. Please login.',
            'user': user_data,
            'redirect': '/login/'
        }, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def login_view(request):
    """
    Login an admin user.
    
    GET /api/login/ - Redirects to login page
    POST /api/login/
    Body: {
        "email": "admin@example.com" or "username": "admin",
        "password": "password123"
    }
    """
    if request.method == 'GET':
        from django.shortcuts import redirect
        return redirect('login_page')
    
    serializer = UserLoginSerializer(data=request.data, context={'request': request})
    
    if serializer.is_valid():
        user = serializer.validated_data['user']
        login(request, user)
        
        # Return user info with redirect to dashboard
        user_data = UserSerializer(user).data
        return Response({
            'message': 'Login successful',
            'user': user_data,
            'redirect': '/dashboard/'  # Redirect to Streamlit dashboard
        }, status=status.HTTP_200_OK)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Logout the current user.
    
    POST /api/logout/
    Requires authentication.
    """
    logout(request)
    return Response({'message': 'Logout successful'}, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """
    Get current authenticated user information.
    
    GET /api/current-user/
    Requires authentication.
    """
    user_data = UserSerializer(request.user).data
    return Response({'user': user_data}, status=status.HTTP_200_OK)


# ------------------------------------------------------------------
# 6. FRONTEND VIEWS (Login & Register Pages)
# ------------------------------------------------------------------

from django.shortcuts import render, redirect

def login_page(request):
    """Render login page - always accessible"""
    # Always show the login page, even if user is authenticated
    # The frontend can show a message if already logged in
    return render(request, 'surveillance_app/login.html')

def register_page(request):
    """Render register page - always accessible"""
    # Always show the register page, even if user is authenticated
    # The frontend can show a message if already logged in
    return render(request, 'surveillance_app/register.html')