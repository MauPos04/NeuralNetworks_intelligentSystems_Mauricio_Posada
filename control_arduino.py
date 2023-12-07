import serial
import time

def read_position(ser):
    # Read the position from the serial port
    line = ser.readline().decode('utf-8').strip()  # assuming data is terminated by a newline
    try:
        position = float(line)
        return position
    except ValueError:
        print("Received invalid data:", line)
        return None

def calculate_control_output(position):
    # Implement your control logic here.
    # For this example, I'm just implementing a dummy control that tries to maintain the position at 50%.
    set_point = 50.0
    error = set_point - position
    control_output = error * 0.1  # simple proportional control
    return control_output

def main():
    # Set up the serial connection
    ser = serial.Serial('COM3', 9600)  # replace 'COM3' with your port and 9600 with your baud rate
    time.sleep(2)  # give some time for the connection to establish
    
    try:
        while True:
            position = read_position(ser)
            if position is not None:
                print(f"Current Position: {position}%")
                control_output = calculate_control_output(position)
                print(f"Sending Control Output: {control_output}")
                ser.write(str(control_output).encode('utf-8'))  # send the control output
                time.sleep(0.1)  # delay for stability, adjust as necessary
    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()