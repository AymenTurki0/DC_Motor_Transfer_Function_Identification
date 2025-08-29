import serial
import csv

ser = serial.Serial("COM13", 115200)  # adjust COM port
with open("encoder_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    while True:
        line = ser.readline().decode().strip()
        if line:
            writer.writerow(line.split(","))
