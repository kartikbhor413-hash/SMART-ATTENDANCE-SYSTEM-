# Smart Attendance System

A **Smart Attendance System** is a web-based attendance management application designed to automate student attendance using **Face Recognition**. The system detects a student's face through a camera, identifies the registered student, and automatically marks their attendance.

The project provides an **ERP-style dashboard** for managing students, attendance records, subjects, and reports. Student face/identity data can be stored locally, while attendance records are maintained digitally for easy tracking and management.

## 🚀 Key Features

- 👤 Student Registration & Management
- 📸 Real-time Face Detection
- 🧠 Face Recognition-Based Attendance
- ✅ Automatic Attendance Marking
- 📊 Attendance Dashboard & Statistics
- 📚 Subject-wise Attendance
- 📅 Date-wise Attendance Records
- 🔍 Student Search & Filtering
- 📈 Attendance Reports
- 💾 Digital Attendance Data Storage
- 🖥️ ERP-style Responsive Web Interface
- 🔐 Admin/Staff Management
- ⚡ Fast and User-Friendly Interface

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **OpenCV**
- **Face Recognition**
- **NumPy**
- **HTML5**
- **CSS3**
- **JavaScript**
- **JSON / Database Storage**

## ⚙️ How It Works

1. Admin/Staff registers students in the system.
2. Student face data is captured and stored.
3. The camera scans the student's face.
4. The system compares the detected face with registered face data.
5. If the student is recognized, attendance is automatically marked.
6. Attendance information is stored with the **student ID, subject, date, and time**.
7. Admin/Staff can view attendance through the dashboard and generate reports.

## 📂 Main Modules

```text
Smart-Attendance-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── students.json
│   ├── attendance.json
│   └── face_data/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── students.html
│   └── attendance.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── reports/
```

## 🎯 Objective

The main objective of this project is to replace traditional manual attendance methods with an **automated, accurate, and efficient attendance management system** using computer vision and web technologies.

## 🔮 Future Enhancements

- Cloud database integration
- Mobile application
- QR-code attendance
- Advanced AI-based face recognition
- Email/SMS notifications
- Student and parent login
- Attendance percentage calculation
- Automatic shortage alerts
- Excel/PDF report generation
- Multi-classroom and multi-department support

## 📌 Project Status

**🚧 Currently under development**

This project is being developed as an **AI/ML + Web Development academic project** with the goal of creating a practical and scalable smart attendance solution.
