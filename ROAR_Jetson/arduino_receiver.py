try:
    from ROAR_Jetson.part import Part
except:
    from part import Part
from serial import Serial
from typing import Optional


class ArduinoCommandReceiver(Part):
    def __init__(self, s: Serial):
        super().__init__(name="ArduinoCMDReceiver")
        self.serial = s
        self.throttle: Optional[float] = None
        self.steering: Optional[float] = None

    def run_step(self):
        if self.serial is not None:
            try:
                vel_wheel = self.serial.readline().decode().rstrip()
                rpm_throttle, rpm_steering = vel_wheel.split(",") # (XXX, YYY)
                self.throttle = float(rpm_throttle[1:])
                self.steering = float(rpm_steering[:-1])
            except Exception as e:
                self.logger.error(e)

    def shutdown(self):
        try:
            self.serial.close()
        except Exception as e:
            self.logger.error(e)


if __name__ == "__main__":
    serial = Serial(port="COM6",
                    baudrate=9600,
                    timeout=0.5,
                    writeTimeout=0.5)
    receiver = ArduinoCommandReceiver(serial=serial)
    while True:
        receiver.run_step()

