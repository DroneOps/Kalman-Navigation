from djitellopy import Tello
tello = Tello()
tello.connect()
tello.streamon()
frame = tello.get_frame_read().frame
print(f"Resolución actual: {frame.shape[1]}x{frame.shape[0]}") # Debería ser 960x720