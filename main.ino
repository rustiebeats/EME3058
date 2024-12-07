#include <SPI.h>
#include <MFRC522.h>

#define RST_PIN 9  // Reset Pin 
#define SS_PIN 10  // Slave Select Pin
#define LED_PIN 7  // LED Pin

MFRC522 rfid(SS_PIN, RST_PIN);

void setup() {
  Serial.begin(9600);
  SPI.begin();
  rfid.PCD_Init();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.println("RFID reader reset");
}

void loop() {
  // 새로운 카드가 감지되지 않으면 대기
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial())
    return;

  Serial.println("카드 감지됨");
  digitalWrite(LED_PIN, HIGH); 
  delay(1000);
  digitalWrite(LED_PIN, LOW);

  rfid.PICC_HaltA();
}