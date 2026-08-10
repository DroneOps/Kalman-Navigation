import Control_Aruco
import threading
from djitellopy import Tello
import time

class MissionControl:
    # Inicialización del tello y camara
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.cap = self.tello.get_frame_read()
        
        self.aruco_controller = Control_Aruco.ArucoController(self, self.cap) # Guarda el codigo del control del aruco
        self.lock = threading.Lock() # Bloquea ciertas acciones para que no se ejecuten al mismo tiempo sino una a la vez
        self.running = True # Nos permite tener activa una accion
        
    # Obtener velocidades
    def get_speeds(self):
        speed_x = self.tello.get_speed_x()
        speed_y = self.tello.get_speed_y()
        speed_z = self.tello.get_speed_z()
        return speed_x, speed_y, speed_z
    # despegar
    def takeoff(self):
        self.tello.takeoff()
    # landear
    def land(self):
        self.tello.land()
    # Guarda constantemente las velocidades
    def save_speeds(self):
        while self.running:
            try:
                _ = self.tello.get_speed_x()
                _ = self.tello.get_speed_y()
                _ = self.tello.get_speed_z()
                time.sleep(0.1)
            except:
                time.sleep(0.5)
    # Comienza el programa de aruco
    def start_mission(self):
        try:
            self.aruco_controller.run()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.land()
            except:
                pass
            try:
                self.tello.streamoff()
                self.tello.end()
            except:
                pass
    
    def send_control_signals(self, control_signal_y, control_signal_z, control_signal_x):
        # Mapeo exacto: control_signal_y (Alineación Izq/Der), control_signal_x (Avance), control_signal_z (Altura)
        self.tello.send_rc_control(int(control_signal_y), int(control_signal_x), int(control_signal_z), 0)

if __name__ == "__main__":
    mission_control = MissionControl()
    
    t1 = threading.Thread(target=mission_control.start_mission)
    t2 = threading.Thread(target=mission_control.save_speeds)
    
    t1.daemon = True
    t2.daemon = True

    try:
        t1.start()
        t2.start()
        
        while t1.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        mission_control.running = False
        try:
            mission_control.tello.send_rc_control(0, 0, 0, 0)
            mission_control.land()
        except:
            pass
    finally:
        try:
            mission_control.tello.streamoff()
            mission_control.tello.end()
        except:
            pass
