import os
import sys
import json
import hashlib
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, static_folder="static", static_url_path="")

# ─── File paths ───────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
STUDENTS_FILE     = os.path.join(DATA_DIR, "students.json")
ACCOUNTS_FILE     = os.path.join(DATA_DIR, "accounts.json")
HISTORY_FILE      = os.path.join(DATA_DIR, "history.json")
TEACHERS_FILE     = os.path.join(DATA_DIR, "teachers.json")
ASSIGNMENTS_FILE  = os.path.join(DATA_DIR, "assignments.json")
FACE_DATA_FILE    = os.path.join(DATA_DIR, "face_data.json")
ATTENDANCE_FILE   = os.path.join(DATA_DIR, "attendance.json")
SESSIONS_FILE     = os.path.join(DATA_DIR, "sessions.json")
ATT_SESSIONS_FILE = os.path.join(DATA_DIR, "attendance_sessions.json")
TIMETABLE_FILE    = os.path.join(DATA_DIR, "timetable.json")
SUBJECTS_FILE     = os.path.join(DATA_DIR, "subjects.json")
SETTINGS_FILE     = os.path.join(DATA_DIR, "settings.json")
REPORTS_DIR       = os.path.join(BASE_DIR, "reports")
MODELS_DIR        = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Computer Vision & Face Recognition Engine ────────────────────
import base64
import cv2
import numpy as np

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")
HAAR_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

yunet_detector = None
sface_recognizer = None
haar_detector = None

try:
    if os.path.exists(YUNET_PATH):
        yunet_detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.55, 0.3, 5000)
except Exception as e:
    print(f"YuNet warning: {e}")

try:
    if os.path.exists(SFACE_PATH):
        sface_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
except Exception as e:
    print(f"SFace warning: {e}")

try:
    if os.path.exists(HAAR_PATH):
        haar_detector = cv2.CascadeClassifier(HAAR_PATH)
except Exception as e:
    print(f"Haar warning: {e}")

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = hashlib.sha256("Admin@123".encode()).hexdigest()

COLLEGE_INFO = {
    "name"    : "Sandip University, Nashik",
    "address" : "Nashik-Pune Highway, Sijul, Mahiravani, Nashik - 422213, Maharashtra, India",
    "phone"   : "+91-2550-670 000",
    "email"   : "info@sandipuniversity.com",
    "website" : "www.sandipuniversity.com",
}

# Data Helpers
def load_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def now_str():
    return datetime.now().strftime("%d-%m-%Y  %I:%M:%S %p")

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def is_strong(pw: str) -> bool:
    return (len(pw) >= 8
            and re.search(r"[A-Z]", pw)
            and re.search(r"\d", pw)
            and re.search(r"[!@#$%^&*]", pw)) is not None

def log_activity(actor: str, action: str, detail: str = ""):
    history = load_json(HISTORY_FILE)
    if isinstance(history, dict):
        history = []
    history.append({
        "timestamp" : now_str(),
        "actor"     : actor,
        "action"    : action,
        "detail"    : detail,
    })
    save_json(HISTORY_FILE, history)

# ─── Sessions & Attendance Helpers ─────────────────────────────────
def load_sessions():
    data = load_json(SESSIONS_FILE)
    if not data:
        data = load_json(ATT_SESSIONS_FILE)
    return data if isinstance(data, dict) else {}

def save_sessions(data):
    save_json(SESSIONS_FILE, data)
    save_json(ATT_SESSIONS_FILE, data)

def decode_image_from_base64(image_data: str):
    if not image_data:
        return None
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_face_embeddings(img):
    """
    Detects faces in BGR image and extracts 128-d L2-normalized embeddings.
    Returns: list of dicts [{"encoding": [float x 128], "bbox": [x,y,w,h], "confidence": float}]
    """
    if img is None or img.size == 0:
        return []

    h, w = img.shape[:2]
    faces_found = []

    # 1. Try YuNet deep learning detector
    if yunet_detector is not None and sface_recognizer is not None:
        try:
            yunet_detector.setInputSize((w, h))
            _, detections = yunet_detector.detect(img)
            if detections is not None and len(detections) > 0:
                for det in detections:
                    score = float(det[14])
                    if score < 0.45:
                        continue
                    aligned = sface_recognizer.alignCrop(img, det)
                    feat = sface_recognizer.feature(aligned).flatten()
                    norm = float(np.linalg.norm(feat))
                    if norm > 0:
                        feat = feat / norm
                    bbox = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                    faces_found.append({
                        "encoding": [round(float(v), 6) for v in feat],
                        "bbox": bbox,
                        "confidence": round(score, 3)
                    })
        except Exception as e:
            print(f"YuNet detect error: {e}")

    # 2. Fallback to Haar Cascade if no face found by YuNet
    if len(faces_found) == 0 and haar_detector is not None and sface_recognizer is not None:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            haar_faces = haar_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            for (x, y, fw, fh) in haar_faces:
                pad = int(fw * 0.08)
                y1 = max(0, y - pad)
                y2 = min(h, y + fh + pad)
                x1 = max(0, x - pad)
                x2 = min(w, x + fw + pad)
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    resized = cv2.resize(crop, (112, 112))
                    feat = sface_recognizer.feature(resized).flatten()
                    norm = float(np.linalg.norm(feat))
                    if norm > 0:
                        feat = feat / norm
                    faces_found.append({
                        "encoding": [round(float(v), 6) for v in feat],
                        "bbox": [int(x), int(y), int(fw), int(fh)],
                        "confidence": 0.90
                    })
        except Exception as e:
            print(f"Haar detect error: {e}")

    return faces_found

def match_encoding_to_student(query_encoding, face_data, students):
    """
    Compares query 128-d encoding with registered students in face_data.json.
    Returns (student_record, min_dist, is_match, best_prn)
    """
    if not query_encoding or len(query_encoding) != 128:
        return None, float("inf"), False, None

    q = np.array(query_encoding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    settings = load_json(SETTINGS_FILE)
    threshold = float(settings.get("face_match_threshold", 0.65))

    best_prn = None
    min_dist = float("inf")
    best_sim = -1.0

    for prn, data in face_data.items():
        ref_enc = data.get("face_encoding") or data.get("face_id")
        if not ref_enc or len(ref_enc) != 128:
            continue
        v = np.array(ref_enc, dtype=np.float32)
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            v = v / v_norm

        dist = float(np.linalg.norm(q - v))
        sim = float(np.dot(q, v))

        if dist < min_dist:
            min_dist = dist
            best_sim = sim
            best_prn = prn

    is_match = False
    if best_prn:
        if min_dist <= threshold or best_sim >= 0.78:
            is_match = True

    if is_match and best_prn in students:
        student = dict(students[best_prn])
        student["prn"] = best_prn
        student["student_id"] = best_prn
        return student, min_dist, True, best_prn

    return None, min_dist, False, best_prn

def generate_next_attendance_id(attendance_records):
    max_num = 0
    for k in attendance_records.keys():
        m = re.search(r"\d+", k)
        if m:
            num = int(m.group())
            if num > max_num:
                max_num = num
    return f"ATT{max_num + 1:06d}"

def check_duplicate_attendance(attendance_records, student_id, session_id, date_str=None, subject=None, lecture_time=None):
    for att in attendance_records.values():
        st_id = att.get("student_id") or att.get("student_prn")
        if st_id != student_id:
            continue
        if session_id and att.get("session_id") == session_id:
            return True, att
        if date_str and (att.get("subject") == subject or att.get("subject_id") == subject) and att.get("lecture_time") == lecture_time:
            if att.get("date") == date_str:
                return True, att
    return False, None
   

# ─── API Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/info")
def get_info():
    return jsonify({"status": "success", "info": COLLEGE_INFO})

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"}), 400

    # Check admin
    if username == ADMIN_USERNAME and hash_pw(password) == ADMIN_PASSWORD:
        log_activity("ADMIN", "ADMIN_LOGIN", "Web Portal Login")
        return jsonify({
            "status": "success",
            "user": {
                "username": "admin",
                "name": "System Administrator",
                "role": "admin"
            }
        })

    accounts = load_json(ACCOUNTS_FILE)
    account = accounts.get(username)

    if not account or account.get("password") != hash_pw(password):
        log_activity(username or "UNKNOWN", "FAILED_LOGIN", "Web Login Attempt")
        return jsonify({"status": "error", "message": "Invalid username or password"}), 401

    role = account.get("role")
    if role == "student":
        students = load_json(STUDENTS_FILE)
        prn = account.get("prn")
        s = students.get(prn, {})
        log_activity(username, "STUDENT_LOGIN", f"PRN: {prn}")
        return jsonify({
            "status": "success",
            "user": {
                "username": username,
                "role": "student",
                "prn": prn,
                "profile": s
            }
        })

    elif role == "teacher":
        teachers = load_json(TEACHERS_FILE)
        tid = account.get("tid")
        t = teachers.get(tid, {})
        log_activity(username, "TEACHER_LOGIN", f"TID: {tid}")
        return jsonify({
            "status": "success",
            "user": {
                "username": username,
                "role": "teacher",
                "tid": tid,
                "profile": t
            }
        })

    return jsonify({"status": "error", "message": "Invalid role"}), 400

@app.route("/api/student/signup", methods=["POST"])
def student_signup():
    data = request.json or {}
    prn = str(data.get("prn", "")).strip().upper()
    name = str(data.get("name", "")).strip().title()
    dob = str(data.get("dob", "")).strip()
    gender = str(data.get("gender", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    phone = str(data.get("phone", "")).strip()
    dept = str(data.get("dept", "")).strip()
    programme = str(data.get("programme", "")).strip()
    year_sem = str(data.get("year_sem", "")).strip()
    address = str(data.get("address", "")).strip()
    guardian = str(data.get("guardian", "")).strip().title()
    guardian_phone = str(data.get("guardian_phone", "")).strip()
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not re.match(r"^[A-Z0-9]{6,15}$", prn):
        return jsonify({"status": "error", "message": "Invalid PRN format (6-15 uppercase alphanumeric characters)"}), 400

    students = load_json(STUDENTS_FILE)
    accounts = load_json(ACCOUNTS_FILE)

    if prn in students:
        return jsonify({"status": "error", "message": "PRN already registered. Please login."}), 400

    if username in accounts or username == ADMIN_USERNAME:
        return jsonify({"status": "error", "message": "Username already taken."}), 400

    if not is_strong(password):
        return jsonify({"status": "error", "message": "Weak password. Must be min 8 chars with 1 uppercase, 1 digit, and 1 special char."}), 400

    ts = now_str()
    student_data = {
        "prn": prn, "name": name, "dob": dob, "gender": gender,
        "email": email, "phone": phone, "dept": dept,
        "programme": programme, "year_sem": year_sem, "address": address,
        "guardian": guardian, "guardian_phone": guardian_phone,
        "username": username, "registered_on": ts
    }

    students[prn] = student_data
    save_json(STUDENTS_FILE, students)

    accounts[username] = {"prn": prn, "password": hash_pw(password), "role": "student"}
    save_json(ACCOUNTS_FILE, accounts)

    # Save text report file
    report_path = os.path.join(REPORTS_DIR, f"{prn}_registration.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  {COLLEGE_INFO['name']}\n")
        f.write(f"  {COLLEGE_INFO['address']}\n")
        f.write("  STUDENT REGISTRATION CONFIRMATION\n")
        f.write("=" * 60 + "\n")
        for k, v in student_data.items():
            if k != "username":
                f.write(f"  {k:<18}: {v}\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Report generated: {ts}\n")

    log_activity(username, "SIGNUP", f"New student registered – PRN: {prn}, Name: {name}")

    return jsonify({
        "status": "success",
        "message": "Registration successful!",
        "prn": prn,
        "student": student_data
    })

@app.route("/api/students", methods=["GET"])
def get_students():
    dept = request.args.get("dept")
    students = load_json(STUDENTS_FILE)
    if dept and dept != "All":
        filtered = {p: s for p, s in students.items() if s.get("dept") == dept}
        return jsonify({"status": "success", "students": filtered})
    return jsonify({"status": "success", "students": students})

@app.route("/api/students/<prn>", methods=["DELETE"])
def delete_student(prn):
    students = load_json(STUDENTS_FILE)
    accounts = load_json(ACCOUNTS_FILE)

    s = students.get(prn)
    if not s:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    username = s.get("username", "")
    del students[prn]
    if username in accounts:
        del accounts[username]

    save_json(STUDENTS_FILE, students)
    save_json(ACCOUNTS_FILE, accounts)

    log_activity("ADMIN", "DELETE_STUDENT", f"Deleted PRN: {prn}, Name: {s.get('name')}")
    return jsonify({"status": "success", "message": f"Student {prn} deleted successfully"})

@app.route("/api/teachers", methods=["GET", "POST"])
def manage_teachers():
    teachers = load_json(TEACHERS_FILE)
    accounts = load_json(ACCOUNTS_FILE)

    if request.method == "GET":
        return jsonify({"status": "success", "teachers": teachers})

    elif request.method == "POST":
        data = request.json or {}
        tid = str(data.get("tid", "")).strip().upper()
        name = str(data.get("name", "")).strip().title()
        email = str(data.get("email", "")).strip().lower()
        phone = str(data.get("phone", "")).strip()
        dept = str(data.get("dept", "")).strip()
        subjects = str(data.get("subjects", "")).strip()
        qual = str(data.get("qual", "")).strip()
        username = str(data.get("username", "")).strip().lower()
        password = str(data.get("password", "")).strip()

        if tid in teachers:
            return jsonify({"status": "error", "message": "Teacher ID already exists."}), 400

        if username in accounts or username == ADMIN_USERNAME:
            return jsonify({"status": "error", "message": "Username already taken."}), 400

        if not is_strong(password):
            return jsonify({"status": "error", "message": "Weak password."}), 400

        ts = now_str()
        teacher_data = {
            "tid": tid, "name": name, "email": email, "phone": phone,
            "dept": dept, "subjects": subjects, "qual": qual,
            "username": username, "created_on": ts, "created_by": "ADMIN"
        }

        teachers[tid] = teacher_data
        save_json(TEACHERS_FILE, teachers)

        accounts[username] = {"tid": tid, "password": hash_pw(password), "role": "teacher"}
        save_json(ACCOUNTS_FILE, accounts)

        log_activity("ADMIN", "ADD_TEACHER", f"TID: {tid}, Name: {name}, Username: {username}")
        return jsonify({"status": "success", "message": "Teacher account created", "teacher": teacher_data})

@app.route("/api/teachers/<tid>", methods=["DELETE"])
def delete_teacher(tid):
    teachers = load_json(TEACHERS_FILE)
    accounts = load_json(ACCOUNTS_FILE)

    t = teachers.get(tid)
    if not t:
        return jsonify({"status": "error", "message": "Teacher not found"}), 404

    username = t.get("username", "")
    del teachers[tid]
    if username in accounts:
        del accounts[username]

    save_json(TEACHERS_FILE, teachers)
    save_json(ACCOUNTS_FILE, accounts)

    log_activity("ADMIN", "DELETE_TEACHER", f"Deleted TID: {tid}, Name: {t.get('name')}")
    return jsonify({"status": "success", "message": f"Teacher {tid} deleted"})

@app.route("/api/teachers/reset-password", methods=["POST"])
def reset_teacher_password():
    data = request.json or {}
    tid = data.get("tid")
    new_pw = data.get("new_password", "")

    teachers = load_json(TEACHERS_FILE)
    accounts = load_json(ACCOUNTS_FILE)

    t = teachers.get(tid)
    if not t:
        return jsonify({"status": "error", "message": "Teacher not found"}), 404

    if not is_strong(new_pw):
        return jsonify({"status": "error", "message": "Weak password"}), 400

    username = t.get("username")
    if username in accounts:
        accounts[username]["password"] = hash_pw(new_pw)
        save_json(ACCOUNTS_FILE, accounts)
        log_activity("ADMIN", "RESET_TEACHER_PASSWORD", f"TID: {tid}")
        return jsonify({"status": "success", "message": f"Password reset for {t.get('name')}"})

    return jsonify({"status": "error", "message": "Account not found"}), 400

@app.route("/api/assignments", methods=["GET", "POST"])
def manage_assignments():
    assignments = load_json(ASSIGNMENTS_FILE)

    if request.method == "GET":
        dept = request.args.get("dept")
        username = request.args.get("teacher_username")

        if username:
            mine = {aid: a for aid, a in assignments.items() if a.get("teacher_username") == username}
            return jsonify({"status": "success", "assignments": mine})

        if dept:
            relevant = {
                aid: a for aid, a in assignments.items()
                if a.get("dept") == dept or a.get("dept") == "All Departments"
            }
            return jsonify({"status": "success", "assignments": relevant})

        return jsonify({"status": "success", "assignments": assignments})

    elif request.method == "POST":
        data = request.json or {}
        aid = f"ASG{datetime.now().strftime('%Y%m%d%H%M%S')}"
        title = data.get("title")
        subject = data.get("subject")
        dept = data.get("dept")
        year_sem = data.get("year_sem")
        due_date = data.get("due_date")
        max_marks = data.get("max_marks", 100)
        description = data.get("description", "")
        teacher = data.get("teacher", "Teacher")
        teacher_username = data.get("teacher_username", "")

        new_assignment = {
            "aid": aid,
            "title": title,
            "subject": subject,
            "dept": dept,
            "year_sem": year_sem,
            "due_date": due_date,
            "max_marks": max_marks,
            "description": description,
            "teacher": teacher,
            "teacher_username": teacher_username,
            "posted_on": now_str()
        }

        assignments[aid] = new_assignment
        save_json(ASSIGNMENTS_FILE, assignments)
        log_activity(teacher_username, "POST_ASSIGNMENT", f"AID: {aid}, Title: {title}")
        return jsonify({"status": "success", "assignment": new_assignment})

@app.route("/api/assignments/<aid>", methods=["PUT", "DELETE"])
def update_assignment(aid):
    assignments = load_json(ASSIGNMENTS_FILE)
    if aid not in assignments:
        return jsonify({"status": "error", "message": "Assignment not found"}), 404

    if request.method == "PUT":
        data = request.json or {}
        a = assignments[aid]
        if "title" in data: a["title"] = data["title"]
        if "subject" in data: a["subject"] = data["subject"]
        if "due_date" in data: a["due_date"] = data["due_date"]
        if "max_marks" in data: a["max_marks"] = int(data["max_marks"])
        if "description" in data: a["description"] = data["description"]
        a["last_edited"] = now_str()

        assignments[aid] = a
        save_json(ASSIGNMENTS_FILE, assignments)
        log_activity(data.get("teacher_username", "TEACHER"), "EDIT_ASSIGNMENT", f"AID: {aid}")
        return jsonify({"status": "success", "assignment": a})

    elif request.method == "DELETE":
        a = assignments.pop(aid)
        save_json(ASSIGNMENTS_FILE, assignments)
        log_activity(request.args.get("username", "TEACHER"), "DELETE_ASSIGNMENT", f"AID: {aid}")
        return jsonify({"status": "success", "message": "Assignment deleted"})

@app.route("/api/stats")
def get_stats():
    students = load_json(STUDENTS_FILE)
    teachers = load_json(TEACHERS_FILE)
    assignments = load_json(ASSIGNMENTS_FILE)

    dept_count = {}
    gender_count = {}

    for s in students.values():
        d = s.get("dept", "Unknown")
        g = s.get("gender", "Unknown")
        dept_count[d] = dept_count.get(d, 0) + 1
        gender_count[g] = gender_count.get(g, 0) + 1

    return jsonify({
        "status": "success",
        "total_students": len(students),
        "total_teachers": len(teachers),
        "total_assignments": len(assignments),
        "by_dept": dept_count,
        "by_gender": gender_count
    })

@app.route("/api/history")
def get_history():
    history = load_json(HISTORY_FILE) or []
    return jsonify({"status": "success", "history": list(reversed(history[-100:]))})

@app.route("/api/auth/change-password", methods=["POST"])
def change_password():
    data = request.json or {}
    username = data.get("username")
    old_pw = data.get("old_password")
    new_pw = data.get("new_password")

    accounts = load_json(ACCOUNTS_FILE)
    account = accounts.get(username)

    if not account or account.get("password") != hash_pw(old_pw):
        return jsonify({"status": "error", "message": "Incorrect current password"}), 400

    if not is_strong(new_pw):
        return jsonify({"status": "error", "message": "Weak new password"}), 400

    accounts[username]["password"] = hash_pw(new_pw)
    save_json(ACCOUNTS_FILE, accounts)

    log_activity(username, "CHANGE_PASSWORD")
    return jsonify({"status": "success", "message": "Password changed successfully"})

# ─── Subjects & Timetable Endpoints ─────────────────────────────────

@app.route("/api/subjects", methods=["GET"])
def get_subjects():
    subjects = load_json(SUBJECTS_FILE)
    return jsonify({"status": "success", "subjects": subjects})

@app.route("/api/timetable", methods=["GET"])
def get_timetable():
    timetable = load_json(TIMETABLE_FILE)
    return jsonify({"status": "success", "timetable": timetable})

# ─── Face ID & Attendance Management API ────────────────────────────

@app.route("/api/face/status", methods=["GET"])
def get_face_status():
    """Returns enrollment status for students without exposing raw face encodings."""
    students = load_json(STUDENTS_FILE)
    face_data = load_json(FACE_DATA_FILE)
    status_map = {}
    for prn, s in students.items():
        fd = face_data.get(prn, {})
        is_reg = bool(fd.get("face_encoding") or fd.get("face_id"))
        status_map[prn] = {
            "prn": prn,
            "name": s.get("name", "Student"),
            "dept": s.get("dept", "Unknown"),
            "is_registered": is_reg,
            "registered_at": fd.get("registered_at", None)
        }
    return jsonify({"status": "success", "students": status_map})

@app.route("/api/face/register", methods=["POST"])
def register_face():
    """
    Authorized user registers a student's face:
    1. Select student from ERP.
    2. Student unique ID loaded.
    3. Detect face from image.
    4. Generate 128-d face encoding.
    5. Save in face_data.json against student's ERP ID.
    """
    data = request.json or {}
    student_id = str(data.get("student_id", "")).strip().upper()
    image_data = data.get("image", "")
    operator_role = str(data.get("operator_role", "")).strip().lower()

    # Security check: only authorized users (admin or teacher)
    if operator_role not in ["admin", "teacher"]:
        return jsonify({"status": "error", "message": "Unauthorized: Only faculty or administrators can register Face IDs."}), 403

    students = load_json(STUDENTS_FILE)
    if student_id not in students:
        return jsonify({"status": "error", "message": f"Student ID '{student_id}' does not exist in ERP records."}), 404

    student = students[student_id]
    img = decode_image_from_base64(image_data)
    if img is None:
        return jsonify({"status": "error", "message": "Invalid or missing image data."}), 400

    faces = extract_face_embeddings(img)
    if len(faces) == 0:
        return jsonify({
            "status": "error",
            "message": "No face detected in camera frame. Please ensure your face is well-lit and directly facing the camera."
        }), 400

    encoding = faces[0]["encoding"]
    ts = now_str()
    face_data = load_json(FACE_DATA_FILE)

    face_record = {
        "student_id": student_id,
        "prn": student_id,
        "name": student.get("name", "Student"),
        "dept": student.get("dept", "Unknown"),
        "face_id": encoding,
        "face_encoding": encoding,
        "registered_at": ts
    }

    face_data[student_id] = face_record
    save_json(FACE_DATA_FILE, face_data)

    log_activity(
        operator_role.upper(),
        "REGISTER_FACE_ID",
        f"Face ID registered for student {student_id} ({student.get('name')})"
    )

    return jsonify({
        "status": "success",
        "message": f"Face ID registered successfully for {student.get('name')} ({student_id})",
        "student": {
            "student_id": student_id,
            "prn": student_id,
            "name": student.get("name"),
            "dept": student.get("dept"),
            "registered_at": ts
        },
        "bbox": faces[0].get("bbox")
    })

@app.route("/api/attendance/sessions", methods=["GET", "POST"])
def manage_attendance_sessions():
    sessions = load_sessions()

    if request.method == "GET":
        return jsonify({"status": "success", "sessions": sessions})

    elif request.method == "POST":
        data = request.json or {}
        subject = str(data.get("subject", "")).strip()
        class_name = str(data.get("class_name", "B.Tech Computer Science")).strip()
        division = str(data.get("division", "1st Year")).strip()
        lecture = str(data.get("lecture", "Lecture 1 (09:00-10:00)")).strip()
        lecture_time = str(data.get("lecture_time", "09:00-10:00")).strip()
        date_str = str(data.get("date", datetime.now().strftime("%Y-%m-%d"))).strip()
        teacher = str(data.get("teacher", "Faculty")).strip()
        teacher_id = str(data.get("teacher_id", "TH101")).strip()
        duration = int(data.get("duration", 60))

        # Generate unique session ID
        clean_date = date_str.replace("-", "")
        clean_subj = re.sub(r"[^A-Za-z0-9]", "", subject) or "LEC"
        count = len([s for s in sessions.values() if s.get("date") == date_str and s.get("subject") == subject]) + 1
        session_id = f"ATT-{clean_date}-{clean_subj}-{count:03d}"

        new_session = {
            "session_id": session_id,
            "subject": subject,
            "class_name": class_name,
            "division": division,
            "lecture": lecture,
            "lecture_time": lecture_time,
            "date": date_str,
            "teacher": teacher,
            "teacher_id": teacher_id,
            "duration": duration,
            "is_active": True,
            "created_at": now_str()
        }

        sessions[session_id] = new_session
        save_sessions(sessions)
        log_activity(teacher, "START_ATTENDANCE_SESSION", f"Session: {session_id}, Subject: {subject}")

        return jsonify({"status": "success", "session": new_session})

@app.route("/api/attendance/sessions/<session_id>/end", methods=["POST"])
def end_attendance_session(session_id):
    sessions = load_sessions()
    if session_id in sessions:
        sessions[session_id]["is_active"] = False
        sessions[session_id]["ended_at"] = now_str()
        save_sessions(sessions)
        log_activity("TEACHER", "END_ATTENDANCE_SESSION", f"Session: {session_id}")
        return jsonify({"status": "success", "message": f"Session {session_id} ended."})
    return jsonify({"status": "error", "message": "Session not found."}), 404

@app.route("/api/attendance/session/<session_id>/stats", methods=["GET"])
def get_session_stats(session_id):
    sessions = load_sessions()
    session = sessions.get(session_id)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404

    students = load_json(STUDENTS_FILE)
    attendance = load_json(ATTENDANCE_FILE)

    # Filter students eligible for this class/department
    dept = session.get("division") or session.get("class_name") or ""
    eligible_students = [
        s for s in students.values()
        if (s.get("dept", "").lower() in dept.lower() or "computer" in s.get("dept", "").lower() or not dept)
    ]
    total_students = len(eligible_students) if len(eligible_students) > 0 else len(students)

    # Present students for this session
    present_records = []
    seen_prns = set()
    for att in attendance.values():
        if att.get("session_id") == session_id:
            prn = att.get("student_id") or att.get("student_prn")
            if prn and prn not in seen_prns:
                seen_prns.add(prn)
                present_records.append({
                    "attendance_id": att.get("attendance_id") or att.get("att_id"),
                    "student_id": prn,
                    "student_name": att.get("student_name", ""),
                    "status": "Present",
                    "method": att.get("method") or att.get("marked_by") or "Face Recognition",
                    "marked_at": att.get("marked_at", "")
                })

    present_count = len(seen_prns)
    remaining_count = max(0, total_students - present_count)
    pct = round((present_count / total_students) * 100, 1) if total_students > 0 else 0.0

    return jsonify({
        "status": "success",
        "session": session,
        "total_students": total_students,
        "present_count": present_count,
        "remaining_count": remaining_count,
        "attendance_percentage": pct,
        "present_students": present_records
    })

@app.route("/api/face/recognize", methods=["POST"])
def recognize_face():
    """
    Automatic Face Recognition Attendance Endpoint:
    1. Receives camera frame.
    2. Detects face and generates 128-d encoding.
    3. Matches with face_data.json to find matching ERP Student ID.
    4. Verifies student in students.json.
    5. Checks duplicate attendance protection for current session.
    6. Automatically marks student PRESENT in attendance.json.
    7. Returns real-time attendance payload for Teacher screen.
    """
    data = request.json or {}
    image_data = data.get("image", "")
    session_id = data.get("session_id", "").strip()
    subject = data.get("subject", "").strip()
    lecture_time = data.get("lecture_time", "").strip()
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d")).strip()
    teacher = data.get("teacher", "Faculty").strip()

    img = decode_image_from_base64(image_data)
    if img is None:
        return jsonify({"status": "no_face", "message": "No image frame received."}), 400

    faces = extract_face_embeddings(img)
    if len(faces) == 0:
        return jsonify({
            "status": "no_face",
            "message": "No face detected in camera view."
        })

    face_data = load_json(FACE_DATA_FILE)
    students = load_json(STUDENTS_FILE)
    attendance = load_json(ATTENDANCE_FILE)
    current_time_str = datetime.now().strftime("%I:%M:%S %p")

    # Process detected faces (supports single or multiple faces)
    results = []
    for face in faces:
        encoding = face["encoding"]
        bbox = face["bbox"]
        student_match, dist, is_match, matched_prn = match_encoding_to_student(encoding, face_data, students)

        if not is_match or not student_match:
            results.append({
                "status": "not_recognized",
                "message": "Face not recognized. Please register your Face ID or contact the administrator.",
                "bbox": bbox,
                "distance": round(dist, 4)
            })
            continue

        prn = student_match.get("prn") or matched_prn

        # Duplicate check (Requirement #7)
        is_dup, dup_record = check_duplicate_attendance(attendance, prn, session_id, date_str, subject, lecture_time)
        if is_dup:
            results.append({
                "status": "already_marked",
                "message": "Attendance already marked.",
                "student": student_match,
                "attendance": dup_record,
                "bbox": bbox,
                "current_time": current_time_str
            })
            continue

        # Automatic Present Marking (Requirement #6)
        att_id = generate_next_attendance_id(attendance)
        marked_at = now_str()
        record = {
            "attendance_id": att_id,
            "att_id": att_id,
            "student_id": prn,
            "student_prn": prn,
            "student_name": student_match.get("name", "Student"),
            "subject_id": subject or "General",
            "subject": subject or "General",
            "session_id": session_id or f"ATT-{date_str.replace('-','')}-001",
            "teacher": teacher,
            "date": date_str,
            "lecture_time": lecture_time or "09:00-10:00",
            "status": "Present",
            "method": "Face Recognition",
            "marked_by": "face_recognition",
            "marked_at": marked_at
        }

        attendance[att_id] = record
        save_json(ATTENDANCE_FILE, attendance)
        log_activity(teacher, "MARK_ATTENDANCE", f"Student {prn} ({student_match.get('name')}) marked PRESENT via Face Recognition")

        results.append({
            "status": "marked_present",
            "message": "Attendance marked PRESENT.",
            "student": student_match,
            "attendance": record,
            "bbox": bbox,
            "current_time": current_time_str
        })

    # Return primary result for the dominant face
    primary = results[0]
    primary["all_detections"] = results
    return jsonify(primary)

@app.route("/api/attendance/student/<prn>", methods=["GET"])
def get_student_attendance(prn):
    """
    Returns student attendance profile:
    Total Lectures, Present, Absent, Attendance Percentage,
    Subject breakdown, and chronological lecture logs.
    """
    prn = str(prn).strip().upper()
    students = load_json(STUDENTS_FILE)
    student = students.get(prn)
    if not student:
        return jsonify({"status": "error", "message": "Student not found"}), 404

    attendance = load_json(ATTENDANCE_FILE)
    face_data = load_json(FACE_DATA_FILE)

    # All attendance records for this student
    my_records = []
    for att in attendance.values():
        st_id = att.get("student_id") or att.get("student_prn")
        if st_id == prn:
            my_records.append({
                "attendance_id": att.get("attendance_id") or att.get("att_id"),
                "subject": att.get("subject") or att.get("subject_id") or "General",
                "date": att.get("date", ""),
                "lecture_time": att.get("lecture_time", ""),
                "status": att.get("status", "Present"),
                "method": att.get("method") or att.get("marked_by") or "Face Recognition",
                "teacher": att.get("teacher", "Faculty"),
                "marked_at": att.get("marked_at", "")
            })

    # Sort descending by date and time
    my_records.sort(key=lambda r: (r.get("date", ""), r.get("marked_at", "")), reverse=True)

    present_count = len(my_records)

    # Calculate total distinct lectures conducted across ERP for this class
    distinct_lectures = set()
    subject_total_map = {}
    for att in attendance.values():
        sess = att.get("session_id")
        subj = att.get("subject") or att.get("subject_id") or "General"
        dt = att.get("date")
        lt = att.get("lecture_time")
        key = sess if sess else f"{dt}_{subj}_{lt}"
        distinct_lectures.add(key)
        if subj not in subject_total_map:
            subject_total_map[subj] = set()
        subject_total_map[subj].add(key)

    total_lectures = max(present_count, len(distinct_lectures))
    absent_count = max(0, total_lectures - present_count)
    pct = round((present_count / total_lectures) * 100, 1) if total_lectures > 0 else 100.0

    # Subject breakdown
    subject_summary = {}
    for r in my_records:
        subj = r.get("subject", "General")
        if subj not in subject_summary:
            total_subj = len(subject_total_map.get(subj, []))
            subject_summary[subj] = {
                "subject": subj,
                "present": 0,
                "total": max(1, total_subj)
            }
        subject_summary[subj]["present"] += 1

    for subj, info in subject_summary.items():
        info["percentage"] = round((info["present"] / info["total"]) * 100, 1)

    # Face ID status
    fd = face_data.get(prn, {})
    is_registered = bool(fd.get("face_encoding") or fd.get("face_id"))

    return jsonify({
        "status": "success",
        "student": {
            "prn": prn,
            "name": student.get("name"),
            "dept": student.get("dept"),
            "programme": student.get("programme"),
            "year_sem": student.get("year_sem")
        },
        "stats": {
            "total_lectures": total_lectures,
            "present_count": present_count,
            "absent_count": absent_count,
            "attendance_percentage": pct
        },
        "subject_summary": list(subject_summary.values()),
        "records": my_records,
        "face_id": {
            "is_registered": is_registered,
            "registered_at": fd.get("registered_at", None)
        }
    })

if __name__ == "__main__":
    print("\n  🎓 Starting Sandip University ERP v2 Server...")
    print("  🌐 Open http://localhost:5000 in your web browser\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

