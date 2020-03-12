#define zero_factor 7403317
#define DOUT  A6
#define CLK   A5
#define DEC_POINT  2
#define sensorPin1 1
#define sensorPin2 2
#define sensorPin3 3
#define sensorPin4 4
#include "HX711.h"
#define PI 3.1415926535897932384626433832795

//float k=1.125; // 1.125=5N
float k=1.459; // 1.459=4N
//float k=1.945; // 1.945=3N
//float k=2.7; // 2.7=2N
float N;

float setpoint=0.1;
//float setpoint=0.22;
//float setpoint=0.32;
float sumfsr = 0;
int numresist=1;
int stageresist=1;
float resist1,resist2,resist3,resist4,resist5,resist6,resist7,resist8,resist9,resist10,meanresist;

//------------ Velocity_guide_motor variable ----------------------

float counter = 0; //This variable will increase or decrease depending on the rotation of encoder
float counter2 = 0; //This variable will increase or decrease depending on the rotation of encoder
float speedMIN1 = 100;

float rps =0,rps1=0;
float velocity=0;// - mean variable of guide
float velocity1=0;// 1 mean variable of handle 
float distance1=0;
float accelaretion=0;
float accelaretion1=0;

unsigned long t = 0;
unsigned long t2 = 0;

//-------------- Motor variable --------------
int Motor_A_Enable = 10;
int Motor_A_Forward = 12; // int1
int Motor_A_Backward = 13; // int2

int RPWM_Output = 6; // Arduino PWM output pin 5; connect to IBT-2 pin 1 (RPWM)
int LPWM_Output = 7; // Arduino PWM output pin 6; connect to IBT-2 pin 2 (LPWM)

float speedMIN;

float friction=0,friction2=0,friction3=0;

//--------------- load cell + buzzer + FSR + limitswitch variable -------------
float calibration_factor =321530.00; 
//float offset=-0.02;
float offset=-0.015;
//float offset=-0.005;
int buzzerPin = 11; 
float val1 = 0;
float val2 = 0;
float val3 = 0;
float val4 = 0;
float val5 = 0;

float force = 0;
float preforce =0;
float postforce=0;
int forcestate=0;

int limit1 = 0;
int limit2 = 0;

float numm = 0;
int num=12;

String data ;
float fsr1Conductance=0,fsr1Voltage=0,fsr1Force=0;
float fsr2Conductance=0,fsr2Voltage=0,fsr2Force=0;
float fsr3Conductance=0,fsr3Voltage=0,fsr3Force=0;
float fsr4Conductance=0,fsr4Voltage=0,fsr4Force=0;

String Stropen= "[";
String Strclose= "]";
String Strlayer= "/";
String Strsplit = ",";
String Strprint1 = Strsplit + 0 + Strsplit + 0 ;

float get_units_kg();

HX711 scale(DOUT, CLK);

void setup() {
  Serial.begin(38400); 

  scale.set_scale(calibration_factor); 
  scale.set_offset(zero_factor); 
  
  pinMode(buzzerPin, OUTPUT);
  pinMode(RPWM_Output, OUTPUT);
  pinMode(LPWM_Output, OUTPUT);
  pinMode(Motor_A_Enable, OUTPUT);
  pinMode(Motor_A_Forward, OUTPUT);
  pinMode(Motor_A_Backward, OUTPUT);
  
  analogWrite(LPWM_Output, 0);
  analogWrite(RPWM_Output, 0);
  digitalWrite(Motor_A_Enable,LOW);
  digitalWrite(Motor_A_Forward,LOW); 
  digitalWrite(Motor_A_Backward,LOW);
  
/*pinMode(2, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 2 
pinMode(3, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 3
pinMode(18, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 18
pinMode(19, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 19*/

pinMode(18, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 18
pinMode(19, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 19
//pinMode(2, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 2
//pinMode(3, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 3
pinMode(20, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 20
pinMode(21, INPUT_PULLUP); // กำหนดให้เป็น pullup input ขา 21
attachInterrupt(5, ai02, RISING); //A rising pulse from encodenren activated ai0(). AttachInterrupt 5 is DigitalPin nr 18 on moust Arduino.
attachInterrupt(4, ai12, RISING); //B rising pulse from encodenren activated ai1(). AttachInterrupt 4 is DigitalPin nr 19 on moust Arduino.
//attachInterrupt(0, ai02, RISING); //A rising pulse from encodenren activated ai02(). AttachInterrupt 0 is DigitalPin nr 2 on moust Arduino.
//attachInterrupt(1, ai12, RISING); //B rising pulse from encodenren activated ai12(). AttachInterrupt 1 is DigitalPin nr 3 on moust Arduino.
attachInterrupt(3, ai0, RISING); //A rising pulse from encodenren activated ai0(). AttachInterrupt 3 is DigitalPin nr 20 on moust Arduino.
attachInterrupt(2, ai1, RISING); //B rising pulse from encodenren activated ai1(). AttachInterrupt 2 is DigitalPin nr 21 on moust Arduino.

 /* pinMode(2, INPUT);
  pinMode(3, INPUT);
  pinMode(18, INPUT);
  pinMode(19, INPUT);
  digitalWrite(2, HIGH); //turn pullup resistor on
  digitalWrite(3, HIGH); //turn pullup resistor on
  digitalWrite(18, HIGH); //turn pullup resistor on
  digitalWrite(19, HIGH); //turn pullup resistor on*/
  
  /*pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);
  pinMode(A6, INPUT);
  digitalWrite(A1, HIGH); //turn pullup resistor on
  digitalWrite(A2, HIGH); //turn pullup resistor on
  digitalWrite(A3, HIGH); //turn pullup resistor on
  digitalWrite(A4, HIGH); //turn pullup resistor on
  digitalWrite(A5, HIGH); //turn pullup resistor on
  digitalWrite(A6, HIGH); //turn pullup resistor on*/
//Setting up interrupt

/*attachInterrupt(0, ai0, RISING); //A rising pulse from encodenren activated ai0(). AttachInterrupt 0 is DigitalPin nr 2 on moust Arduino.
attachInterrupt(1, ai1, RISING); //B rising pulse from encodenren activated ai1(). AttachInterrupt 1 is DigitalPin nr 3 on moust Arduino.
attachInterrupt(5, ai02, RISING); //A rising pulse from encodenren activated ai0(). AttachInterrupt 5 is DigitalPin nr 18 on moust Arduino.
attachInterrupt(4, ai12, RISING); //B rising pulse from encodenren activated ai1(). AttachInterrupt 4 is DigitalPin nr 19 on moust Arduino.*/

}

void loop() {

//---------------------loop for receive FSR + limit Switch value---------------
    val1 = analogRead(A1);
    val2 = analogRead(A2);
    val3 = analogRead(A3);
    val4 = analogRead(A4);
    limit1 =digitalRead(8);
    limit2 =digitalRead(9);
    
fsr1Voltage = map(val1, 0, 1023, 0, 5000);
  fsr1Conductance=1000000/(((5000-fsr1Voltage)*10000)/fsr1Voltage);
  if (fsr1Conductance <= 1000) {
      fsr1Force = fsr1Conductance / 80;    
    } else {
      fsr1Force = fsr1Conductance - 1000;
      fsr1Force /= 30;}

fsr2Voltage = map(val2, 0, 1023, 0, 5000);
  fsr2Conductance=1000000/(((5000-fsr2Voltage)*10000)/fsr2Voltage);
  if (fsr2Conductance <= 1000) {
      fsr2Force = fsr2Conductance / 80;    
    } else {
      fsr2Force = fsr2Conductance - 1000;
      fsr2Force /= 30;}

fsr3Voltage = map(val3, 0, 1023, 0, 5000);
  fsr3Conductance=1000000/(((5000-fsr3Voltage)*10000)/fsr3Voltage);
  if (fsr3Conductance <= 1000) {
      fsr3Force = fsr3Conductance / 80;    
    } else {
      fsr3Force = fsr3Conductance - 1000;
      fsr3Force /= 30;}

fsr4Voltage = map(val4, 0, 1023, 0, 5000);
  fsr4Conductance=1000000/(((5000-fsr4Voltage)*10000)/fsr4Voltage);
  if (fsr4Conductance <= 1000) {
      fsr4Force = fsr4Conductance / 80;    
    } else {
      fsr4Force = fsr4Conductance - 1000;
      fsr4Force /= 30;}

//---------------------loop for receive Force (load cell) value---------------

if(num==12&&forcestate==0){
    String data = String(get_units_kg()+offset, DEC_POINT); 
    postforce = preforce;
    preforce = data.toFloat() ;
    //if(preforce>=0.01&&preforce<=0.01){preforce = 0;}
    //Serial.print(preforce); 
    if( preforce >= 1.38&&preforce <= 1.39&&postforce <= 0.02&&postforce >= -0.02){preforce=preforce-preforce;}
    if( preforce >= 1.38&&preforce <= 1.39&&postforce <= 0){preforce=postforce;}
    forcestate =1;
    num = 0;
  }

if(forcestate =1){
  force=(((12-num)*postforce)+(num*preforce))/12;
  }
  num=num+1;

if(num==13&&forcestate==1){
    String data = String(get_units_kg()+offset, DEC_POINT); 
    preforce = postforce;
    postforce = data.toFloat() ;
    if(postforce>=0.01&&postforce<=0.01){postforce = 0;}
    forcestate =0;
    num = 0;
  }
if(forcestate =0){
    force=(((12-num)*preforce)+(num*postforce))/12;
  }

//--------------------- loop for control speed motor depend on Force ---------------

if(force>=0){
    speedMIN = abs(force);
    speedMIN = trai(speedMIN);
    speedMIN = k*speedMIN;
    if(speedMIN>255){speedMIN=255;}
      if(limit2==1){
        analogWrite(LPWM_Output, 0);
        analogWrite(RPWM_Output, 0);}
      else {analogWrite(LPWM_Output, 0);
            analogWrite(RPWM_Output, speedMIN);}
}
if(force<0){
    speedMIN = abs(force);
    speedMIN = trai(speedMIN);
    speedMIN = k*speedMIN;
    if(speedMIN>255){speedMIN=255;}
      if(limit1==1){
        analogWrite(LPWM_Output, 0);
        analogWrite(RPWM_Output, 0);}
      else {analogWrite(LPWM_Output, speedMIN);
            analogWrite(RPWM_Output, 0);}
}

//--------------------- loop for control speed motor velocity guide and calculate distance , velocity , acceleration ---------------

t = millis();

analogWrite(Motor_A_Enable, speedMIN1);
digitalWrite(Motor_A_Forward, LOW);    
digitalWrite(Motor_A_Backward, HIGH);

if ((unsigned long)(t - t2) >= 10) {
accelaretion = velocity;
accelaretion1 = velocity1;//velocity older;

rps = ((abs(counter)*2*PI*100)/720); // rad/sec
rps1 = ((abs(counter2)*2*PI*100)/720); // rad/sec
velocity = (0.0235*rps);// unit metre per sec.
velocity1 = (0.00611*rps1);//unit metre per sec.
distance1 = distance1 +((counter2*2*PI*0.00611)/720); // unit metre.
  
accelaretion = (abs(velocity-accelaretion))*100;// unit metre per sec^2.
accelaretion1 = (abs(velocity1-accelaretion1))*100;// unit metre per sec^2.
//if(abs(velocity)<=(setpoint-0.04)){speedMIN1=speedMIN1+2;if(speedMIN1>255){speedMIN1=255;}if(speedMIN1<0){speedMIN1=0;}}//adjust speed velocity guide to setpoint.
//if(abs(velocity)>=(setpoint+0.04)){speedMIN1=speedMIN1-2;if(speedMIN1>255){speedMIN1=255;}if(speedMIN1<0){speedMIN1=0;}}

if(abs(velocity)<=(setpoint-0.005)){speedMIN1=speedMIN1+0.1;if(speedMIN1>255){speedMIN1=255;}if(speedMIN1<0){speedMIN1=0;}}//adjust speed velocity guide to setpoint.
if(abs(velocity)>=(setpoint+0.005)){speedMIN1=speedMIN1-0.1;if(speedMIN1>255){speedMIN1=255;}if(speedMIN1<0){speedMIN1=0;}}

counter=0;
counter2=0;

t2 = millis();
}

//--------------get static friction------------------
if(accelaretion1>0&&friction>0&&friction2>0&&friction3==0){friction3=abs(force);} 
if(accelaretion1>0&&friction>0&&friction2==0){friction2=abs(force);}
if(accelaretion1>0&&friction==0){friction=abs(force);}
    
    /*
      Serial.print(accelaretion1);
      Serial.print(','); 
      Serial.print(friction);
      Serial.print(','); 
      Serial.print(friction2);
      Serial.print(','); 
      Serial.print(friction3);
      Serial.print(','); 
      Serial.println(force);*/

//---------------- Print value to serial monitor ----------------
String Strprint = Strsplit + fsr1Force + Strsplit + fsr2Force+ Strsplit + fsr3Force + Strsplit + fsr4Force + Strsplit + force;
    
 //if(numm==1){
 //if(numm==1&&force>-0.1&&force<0.1&&sumfsr>1){
 //if(numm==1&&force>-0.1&&force<0.1){
 //if(numm==1&&force>0.03){
 if(numm==1&&force<-0.03){
 //if(numm==1&&force<-0.05||force>0.05){

      /*Serial.print(friction);
      Serial.print(','); 
      Serial.print(friction2);
      Serial.print(','); 
      Serial.print(friction3);
      Serial.print(','); 
      //Serial.print(velocityold);
      //Serial.print(','); 
      Serial.print(velocitynew);
      Serial.print(',');  */
//      Serial.print(limit1); 
//      Serial.print(limit2); 
//      Serial.print(','); 
      //Serial.print(speedMIN);
      /*Serial.print(num);
      Serial.print(','); 
      Serial.print(preforce);
      Serial.print(','); 
      Serial.print(postforce);
      Serial.print(','); 
      Serial.println(force);*/
      /*Serial.print(accelaretion);
      Serial.print(','); 
      Serial.print(velocity);
      Serial.print(','); 
      Serial.print(speedMIN1);*/
      //Serial.println(force,5);
      //Serial.print(num); 
      
      //Serial.print(','); 
      //Serial.print(postforce);
      //Serial.print(speedMIN1); 
     // Serial.print("Distance : ");Serial.print(distance1);Serial.println("metre");
      Serial.print("force : ");Serial.print(force,3);Serial.println("  newtons");
      
      /*Serial.print(Strprint); 
      Serial.print(','); 
      Serial.print(velocity1);
      Serial.print(','); 
      Serial.println(distance1);*/
      
//      if(velocity1>=0.01&&numresist==1&&stageresist==1){resist1=preforce*9.81; numresist=numresist+1; ;Serial.print("resist1 :");Serial.println(resist1);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==2&&stageresist==1){resist2=preforce*9.81; numresist=numresist+1; Serial.print("resist2 :");Serial.println(resist2);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==3&&stageresist==1){resist3=preforce*9.81; numresist=numresist+1; Serial.print("resist3 :");Serial.println(resist3);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==4&&stageresist==1){resist4=preforce*9.81; numresist=numresist+1; Serial.print("resist4 :");Serial.println(resist4);stageresist=0;}
//      if(velocity1>=0.01&&numresist==5&&stageresist==1){resist5=preforce*9.81; numresist=numresist+1; Serial.print("resist5 :");Serial.println(resist5);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==6&&stageresist==1){resist6=preforce*9.81; numresist=numresist+1; Serial.print("resist6 :");Serial.println(resist6);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==7&&stageresist==1){resist7=preforce*9.81; numresist=numresist+1; Serial.print("resist7 :");Serial.println(resist7);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==8&&stageresist==1){resist8=preforce*9.81; numresist=numresist+1; Serial.print("resist8 :");Serial.println(resist8);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==9&&stageresist==1){resist9=preforce*9.81; numresist=numresist+1; Serial.print("resist9 :");Serial.println(resist9);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==10&&stageresist==1){resist10=preforce*9.81; numresist=numresist+1; Serial.print("resist10 :");Serial.println(resist10);stageresist=0;} 
//      if(velocity1>=0.01&&numresist==11&&stageresist==1){meanresist = (resist1+resist2+resist3+resist4+resist5+resist6+resist7+resist8+resist9+resist10)/10;Serial.print("meanresist :");Serial.println(meanresist);stageresist=0;}
    }
    else {//Serial.print("Distance : ");Serial.print(distance1);Serial.println("  metre");
      Serial.print("force : ");Serial.print(force,3);Serial.println("  newtons");
      //Serial.print(velocity); Serial.print(','); Serial.print(speedMIN1);  Serial.println(Strprint1); 
    }
//     if(numm==1&&force>0.03){stageresist=1;}
    if(fsr1Force>=0&&fsr2Force>=0&&fsr3Force>=0&&fsr4Force>=0&&numm==0){
      /*Serial.print(friction);
      Serial.print(','); 
      Serial.print(friction2);
      Serial.print(','); 
      Serial.print(friction3);
      Serial.print(','); 
      //Serial.print(velocityold);
      //Serial.print(','); 
      Serial.print(velocitynew);
      Serial.print(','); 
      Serial.print(limit1); 
      Serial.print(limit2);
      Serial.print(',');  
      Serial.print(speedMIN);*/
      //Serial.print(speedMIN1);
     /*Serial.print(distance1);
      Serial.print(',');
      Serial.print(velocity);
      Serial.print(','); 
      Serial.print(speedMIN1);*/
      Serial.println(Strprint); 
      Serial.print(','); 
      Serial.println(velocity1);
      
      numm=1;

    }

sumfsr= fsr1Force + fsr2Force + fsr3Force + fsr4Force;
/*float sound = abs(sumfsr)/8;
sound = trai(sound);
   Serial.print(sumfsr);
   Serial.print('\t');
   Serial.println(sound);*/
if (sumfsr >= 1) {
  if (sumfsr >= 1&&sumfsr <= 3){ 
  analogWrite(buzzerPin, 250); }
  if (sumfsr >3&&sumfsr <= 5){ 
  analogWrite(buzzerPin, 100); }
  if (sumfsr > 5&&sumfsr <= 7){ 
  analogWrite(buzzerPin, 0); }
  }
else analogWrite(buzzerPin ,255);

 
}


float get_units_kg()
{
  return(scale.get_units()*0.453592);
}

float trai(float x)
{
  return((x*255)/3);
}
void ai0() {
// ai0 is activated if DigitalPin nr 2 is going from LOW to HIGH
// Check pin 3 to determine the direction
if(digitalRead(21)==LOW) {
counter=counter+1;
}else{
counter=counter-1;
}
}
 
void ai1() {
// ai0 is activated if DigitalPin nr 3 is going from LOW to HIGH
// Check with pin 2 to determine the direction
if(digitalRead(20)==LOW) {
counter=counter-1;
}else{
counter=counter+1;
}
}
void ai02() {
// ai0 is activated if DigitalPin nr 2 is going from LOW to HIGH
// Check pin 3 to determine the direction
if(digitalRead(19)==LOW) {
counter2=counter2+1;
}else{
counter2=counter2-1;
}
}
 
void ai12() {
// ai0 is activated if DigitalPin nr 3 is going from LOW to HIGH
// Check with pin 2 to determine the direction
if(digitalRead(18)==LOW) {
counter2=counter2-1;
}else{
counter2=counter2+1;
}
}
