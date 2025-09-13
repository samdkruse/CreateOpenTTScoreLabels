import cv2

video_path = f"/Users/sdk/Downloads/game_5.mp4"
print(f"\n🎥 {video_path}")
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 8740)
ret, frame = cap.read()
cap.release()


# Select left player's score
print("👤 Select ROI for Player 1 (left player)")
roi1 = cv2.selectROI("Player 1 Score", frame, showCrosshair=True)
cv2.destroyAllWindows()

# Select right player's score
print("👤 Select ROI for Player 2 (right player)")
roi2 = cv2.selectROI("Player 2 Score", frame, showCrosshair=True)
cv2.destroyAllWindows()

(x1, y1, w1, h1) = roi1
(x2, y2, w2, h2) = roi2

print(f"✅ Player 1 ROI: x={x1}, y={y1}, width={w1}, height={h1}")
print(f"✅ Player 2 ROI: x={x2}, y={y2}, width={w2}, height={h2}")
