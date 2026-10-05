src = open('static/app.js', encoding='utf-8').read()
print('JS size:', len(src), 'bytes')
checks = [
    'loadSubjects', 'loadStudentAttendance', 'filterStudentAttendanceTable',
    'initTeacherAttendancePanel', 'startTeacherAttendanceSession',
    'startRecognitionLoop', 'handleRecognitionResult',
    'stopTeacherAttendanceSession', 'openFaceRegistrationModal',
    'captureAndRegisterFace', 'closeFaceRegistrationModal',
    'togglePauseAttendanceCamera', 'playBeep', 'refreshStudentAttendance',
    'onFaceRegStudentSelected', 'startRegCamera', 'stopRegCamera',
    'toggleRegCamera', 'startTeacherCamera', 'stopTeacherCamera',
    'updateLiveSessionStats', 'addStudentToSessionPresentList',
    'drawFaceBoundingBox', 'setRealtimeField', 'populateSubjectDropdowns'
]
all_ok = True
for fn in checks:
    found = fn in src
    status = 'OK' if found else 'MISSING'
    if not found:
        all_ok = False
    print(f'  {fn}: {status}')
print()
print('ALL OK' if all_ok else 'SOME FUNCTIONS MISSING')
