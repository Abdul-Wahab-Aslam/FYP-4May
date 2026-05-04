import cv2

# Aapki original video ka naam
input_video = "Car accident.mp4"
output_video = "processed_demo.mp4"

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video save karne ka setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500: # Sirf bari gariyan detect karega
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

    accident = False
    # Check collision
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[j]
            # Agar gariyan bohot qareeb aa jayen (Accident logic)
            if abs(x1 - x2) < (w1+w2)/2 and abs(y1 - y2) < (h1+h2)/2:
                accident = True

    # Draw Boxes
    for (x, y, w, h) in boxes:
        color = (0, 0, 255) if accident else (0, 255, 0)
        label = "COLLISION RISK" if accident else "Vehicle (EV)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if accident:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
        cv2.putText(frame, "ACCIDENT DETECTED! V2V ALERT SENT", (50, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("Success! 'processed_demo.mp4' ban gayi hai.")