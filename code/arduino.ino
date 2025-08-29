// --- Encoder setup ---
const int encoderPinA = 2;   // Encoder channel A (interrupt)
const int encoderPinB = 3;   // Encoder channel B
volatile long encoderCount = 0;

// --- Motor setup ---
const int motorIn1 = 9;   // IN1 on driver
const int motorIn2 = 10;   // IN2 on driver

// --- Timing setup ---
unsigned long lastTime = 0;
const unsigned long sampleInterval = 10; // ms (100 Hz sampling)

// Encoder specs
const int countsPerRevolution = 350;  // adjust for your encoder

void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(motorIn1, OUTPUT);
  pinMode(motorIn2, OUTPUT);

  // Encoder pins
  pinMode(encoderPinA, INPUT_PULLUP);
  pinMode(encoderPinB, INPUT_PULLUP);

  // Attach interrupt
  attachInterrupt(digitalPinToInterrupt(encoderPinA), encoderISR, RISING);

  // Start motor forward
  analogWrite(motorIn1, 25);
  digitalWrite(motorIn2, LOW);

  // Print CSV header
  Serial.println("time_ms,speed_rpm");
}

void loop() {
  unsigned long now = millis();
  if (now - lastTime >= sampleInterval) {
    lastTime = now;

    // Capture and reset encoder count
    noInterrupts();
    long count = encoderCount;
    encoderCount = 0;
    interrupts();

    // Convert counts → RPM
    float countsPerSec = (count * 1000.0) / sampleInterval;
    float revPerSec = countsPerSec / countsPerRevolution;
    float rpm = revPerSec * 60.0;

    // Print as CSV
    
    Serial.print(now);
    Serial.print(",");
    Serial.println(rpm);
    
  }
}

// --- Encoder ISR ---
void encoderISR() {
  int b = digitalRead(encoderPinB);
  if (b == HIGH) {
    encoderCount++;  // CW
  } else {
    encoderCount--;  // CCW
  }
}
