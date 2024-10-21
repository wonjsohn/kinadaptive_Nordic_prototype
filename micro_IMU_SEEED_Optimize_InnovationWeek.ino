 /* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expr ess or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Author: Won Joon Sohn. wonjoon.sohn@abbott, wonjsohn@gmail
// 20230411 - dual model (speech + imu) realized
// 20230412 - add BLE + LED display. 
// 20230413 - LED display (u8x8) doesn't work with IMU), freezes. 
// 20230416 - Debugging the problems of Tflite and IMU in the board crashing.  
// 20230419 - 'const' declaration for model keeps the large size model in the flash memory. But this hangs in 'invoke' 
//            if 'const' is removed, the program suffers low memory problem b/c the model is stored in heap memory.  
//            alignas(8) const unsigned char model[] works for some reason. Is this b/c Flash is by default block accessed (as opposed to byte addresses?) 

#include <TensorFlowLite.h>

#define IMU_ONLY    true
#define IMU_AND_SPEECH false
#define READ_RAW_SIGNAL true
#define READ_3DOF_ONLY true
#define PAUSE_AFTER_CUE true

#define BLE_ON true
#define OLED_INTERVAL_MS                750
#define OLED_IDLE_TURNOFF_MS            3000
#define OLED_ON true
#define PRINTOUT_ABOVE_SERIALOUT_PLOTTING false

#define STEP_REPORT_INTERVAL_MS        5000  // 5 seconds
#define INSERT_DELAY_AFTER_FREEFALL   true 

/* SSD1306 related - use u8x8 library */
#include <U8x8lib.h>

#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>

// Define pin connections
#define OLED_MOSI  10
#define OLED_CLK   8
#define OLED_DC    7
#define OLED_CS    6
#define OLED_RESET 5
// Software SPI configuration
//8X8_SSD1306_128X64_NONAME_4W_SW_SPI u8x8(OLED_CLK, OLED_MOSI, OLED_CS, OLED_DC, OLED_RESET);

#endif
#ifdef U8X8_HAVE_HW_I2C
U8X8_SSD1306_128X64_NONAME_HW_I2C u8x8(/* reset=*/ U8X8_PIN_NONE);         

#include <Wire.h>
#endif
//U8X8_SSD1306_128X64_NONAME_SW_I2C u8x8(/* clock=*/ 5, /* data=*/ 4, /* reset=*/ U8X8_PIN_NONE);         // Digispark ATTiny85
//U8X8_SSD1306_128X64_ALT0_HW_I2C u8x8(/* reset=*/ U8X8_PIN_NONE);         // same as the NONAME variant, but may solve the "every 2nd line skipped" problem


#if BLE_ON
#include <ArduinoBLE.h>       // BLE library

// BLE Service Name

BLEDevice central;
BLEService IMUService("180C"); // 180C for User defined service?

// BLE Orientation Characteristic
//BLECharacteristic IMUSensorData("2A19",  // standard 16-bit characteristic UUID
//  BLERead | BLENotify, 2); // remote clients will be able to get notifications if this characteristic changes

// BLERead – allows remote devices to read the characteristic value
// BLEWriteWithoutResponse – allows remote devices to write to the device without expecting an acknowledgement
// BLEWrite – allows remote devices to write, while expecting an acknowledgement the write was successful
// BLENotify – allows a remote device to be notified anytime the characteristic’s value is update
// BLEIndicate – the same as BLENotify, but we expect a response from the remote device indicating it read the value
BLEStringCharacteristic IMUSensorData("2A19", BLERead  | BLENotify, 512);

BLEStringCharacteristic rxChar("2A18",  BLEWriteWithoutResponse | BLEWrite, 512); // can be optimized
// Setup the incoming data characteristic (RX).
//const int RX_BUFFER_SIZE = 256;
//bool RX_BUFFER_FIXED_LENGTH = false;



#endif 


// Display related 
static unsigned long last_interval_ms = 0;
static unsigned long last_oled_interval_ms = 0;
static unsigned long last_idle_start_time_ms = 0;
static unsigned long currentMillis = 0;
static unsigned long previousMillis = 0;
static uint8_t count_index=0;
static unsigned long last_rXTime_ms = 0;
static unsigned long last_stepTime_ms = 0;

//time variables
unsigned long previousTime = 0;         // end time of the last period in microseconds
unsigned long currentTime = 0;
unsigned long deltaTime = 0;
const unsigned long period = 10000;     // time period for pulse counting in microseconds

int decoded_state_index = 0;
//bool is_new_gesture = false;
boolean is_new_gesture = false;
boolean is_free_fall_detected = false;
boolean is_stepcount_report = false;
//int decoded_state_index_prev = 0;


#if IMU_ONLY
//#include <SEEED_LSM6DS3.h>
#include "src/LSM6DS3/LSM6DS3.h"// 
#include <Wire.h>

// TEST 0408
//#include "audio_provider.h"
//#include "command_responder.h"
//#include "feature_provider.h"
//#include "main_functions.h"
//#include "micro_features_micro_model_settings.h"
//#include "micro_features_model.h"
//#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"  //Can use all_ops_resolver but with inefficient size
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
// END TEST 0408

//#include <tensorflow/lite/micro/all_ops_resolver.h>
//#include <tensorflow/lite/micro/micro_error_reporter.h>
//#include <tensorflow/lite/micro/micro_interpreter.h>
//#include <tensorflow/lite/schema/schema_generated.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "imu_model.h" 
//#include "epg_model0.h"
//#include "epg_model20240827.h"
//#include "epg_model_20240827_0901.h"
//#include "EPG_20240909_2.h"
#include "EPG_20241017_1018_4class_3dof_tf_keras_N35_N15_25hz.h"


//#include "acc_only_model.h"

//const float accelerationThreshold = 23.0;//(epg_model0827: 20 b/c of 2g setting) 2.5; // (float model) threshold of significant in G's
//const uint16_t accelerationThreshold = 2*16384; // threshold of significant in G's (16384=1g in 4g setting)
const float accelerationThreshold = 16384*2.5; // threshold of significant in G's



const int numSamples = 200;//(epg_model0827: 100Hz x 2s= 200) was 240;  // (now 1s = 120)  1 second = 119.  1.5 second = 179.  

int samplesRead = numSamples;


/* Private variables ------------------------------------------------------- */
LSM6DS3 myIMU(I2C_MODE, 0x6A);

#endif 
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal



/* Interrupt related ------------------------------------------------------- */
//#define int2Pin PIN_LSM6DS3TR_C_INT1 // pins_arduino.h
uint8_t interruptCount = 0; // Amount of received interrupts

bool is_new_speech_command = false;  // common to all three modes


// Globals, used for compatibility with Arduino-style sketches.
namespace {

#if IMU_ONLY
// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
//tflite::AllOpsResolver tflOpsResolver;

const unsigned int tOpCount = 3;
tflite::MicroMutableOpResolver<tOpCount> tflMicroMutableOpsResolver;

const tflite::Model* model_imu = nullptr;  // imu model
tflite::MicroInterpreter* interpreter_imu = nullptr;

TfLiteTensor* model_input4imu = nullptr;
TfLiteTensor* model_output4imu = nullptr; //added for IMU
#endif


// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
#if IMU_ONLY
constexpr int kTensorArenaSize_IMU = 8 * 1024; 
byte tensor_arena_imu[kTensorArenaSize_IMU] __attribute__((aligned(16)));

#endif

}  // namespace

// In case IMU for gesture control is used
#if IMU_ONLY || IMU_AND_SPEECH
// array to map gesture index to a name
const char* BASIC_CLASSES[] = {
  "STEP-BY-STEP",
  "FREE FALL   ",
  "INACTIVE    ",
};


const char* GESTURES[] = {
//  "punch",
//  "flex"
//    "prone  ",
//    "Sit2std ",
//    "Stnd2sit",
//    "supine  ",
//    "turnRt  ",
//    "walking ", 
//	  "Supine  "
//    "prone   ",    //0909
//    "stnd2sit",    //0909
//    "supine  ",    //0909
//    "walk    ",    //0909
    "prone   ",  //1017
    "sideR   ",  //1017
    "stand   ",  //1017
    "supine  ",  //1017
};
const char* BASIC_CLASSES_longtext[] = {
  "STEP-BY-STEP 4mA/100Hz",
  "FREE FALL    ",
  "INACTIVE    1mA/100Hz  ",
};					 

							   

char *GESTURES_longtext[] = {
    "prone   ",    //0909
    "sideR  ",
    "stand   ",
    "supine  ",
//  "TOGGLE      ",
//  "TURN OFF "
};  // re-arrange according to the edge impulse order

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

#endif

// seeed_newmodel //
//char *possible_states[] = {
//  "currentup ", //
//  "currentdown  ", //
//  "IDLE    ",  //
//  "TREMOR  ", //
//  "WAKE UP "  // 
//};  // re-arrange according to the edge impulse order



//This is not used
enum ACTIVITY_STATES{
	STATE_CURRENTUP, 
	STATE_CURRENTDOWN,
	STATE_WAKEUP,
	STATE_TOGGLE,      
	STATE_TURNOFF,

};

const int RED_ledPin =  11;
const int BLUE_ledPin =  12;
const int GREEN_ledPin =  13; 
			  				 			
//free-fall detection related: 
uint16_t detectCount = 0;

//    "yesstim",
//    "forceoff",
//  STATE_STIMON,
//  STATE_VOICE_STIM_OFF,  


    
// The name of this function is important for Arduino compatibility.
void setup() {
	Serial.begin(115200);
	//while (!Serial);



#if BLE_ON
	// begin BLE initialization
	if (!BLE.begin()) {
	Serial.println("starting BLE failed!");

	while (1);
	}

	// ** BLE Setup **
	/* Set a local name for the BLE device
	This name will appear in advertising packets
	and can be used by remote devices to identify this BLE device
	The name can be changed but maybe be truncated based on space left in advertisement packet
	*/

	// set advertised local name and service UUID:
	BLE.setDeviceName("SEEED-WJ");
	BLE.setLocalName("SEEED-WJ");
	BLE.setAdvertisedService(IMUService); // add the service UUID
	IMUService.addCharacteristic(IMUSensorData); // add the IMU characteristic
	IMUService.addCharacteristic(rxChar); // add characteristic  // NEW 0510

	BLE.addService(IMUService); // Adding the service to the BLE stack
	IMUSensorData.writeValue("hello"); // set initial value for this characteristic  : this is showing up in the "battery level" in BLE scanner


	/* NEW 0510- add a callback whhich will fire every time the Bluetooth LE hardware has a characteristic
	 written. This allows us to handle data as it streams in. */
	// Bluetooth LE connection handlers.
	//  BLE.setEventHandler(BLEConnected, onBLEConnected);
	//  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

	// Event driven reads. (call back)
	//rxChar.setEventHandler(BLEWritten, onRxCharValueUpdate);


	/* Start advertising BLE.  It will start continuously transmitting BLE
	 advertising packets and will be visible to remote BLE central devices
	 until it receives a new connection */

	// start advertising
	BLE.advertise();

	Serial.println("Bluetooth device active, waiting for connections...");
	String address = BLE.address();
	Serial.print("Local address is: ");
	Serial.println(address);
#endif //BLE_ON



					
  
  tflMicroMutableOpsResolver.AddFullyConnected();
  tflMicroMutableOpsResolver.AddSoftmax();
  tflMicroMutableOpsResolver.AddRelu();
    

#if IMU_ONLY || IMU_AND_SPEECH


  if (myIMU.begin() != 0) {
      Serial.println("Device error");
  } else {
     Serial.println("Device OK!");
  }

  /* -- some IMU setting --*/ 

  /*--- FREE FALL & PEDOMETER SETTING --*/
  config_free_fall_detect();// fall detect + pedometer test
  //config_activity_inactivity();
  //config_pedometer();
  //pinMode(int2Pin, INPUT);
  //attachInterrupt(digitalPinToInterrupt(int2Pin), int1ISR, RISING);


  Serial.println();
  Serial.println("GESTURES CLASS ORDER:");
  Serial.print(GESTURES[0]);
  Serial.print(",");
  Serial.print(GESTURES[1]);
  Serial.print(",");
  Serial.print(GESTURES[2]);
  Serial.print(",");
  Serial.print(GESTURES[3]);
  Serial.print(",");
//  Serial.print(GESTURES[4]);
//  Serial.println(",");

//  Serial.print(GESTURES[5]);
//  Serial.println(",");


  model_imu = tflite::GetModel(EPG_20241017_1018_4class_3dof_tf_keras_N35_N15_25hz_model);
    if (model_imu->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("model_imu schema mismatch!");
        while (1);
   
    }
#endif // IMU_ONLY || IMU_AND_SPEECH



#if IMU_ONLY || IMU_AND_SPEECH
    // same except 'imu_model' vs 'model'in the intrepreter
    
    // Build an interpreter to run the model with.
    interpreter_imu = new tflite::MicroInterpreter(
        model_imu, tflMicroMutableOpsResolver, tensor_arena_imu, kTensorArenaSize_IMU, &tflErrorReporter);
    
    // Allocate memory from the tensor_arena for the model's tensors.
    interpreter_imu->AllocateTensors();
    
    // Get information about the memory area to use for the model's input.
    model_input4imu = interpreter_imu->input(0);
    model_output4imu = interpreter_imu->output(0);

#endif //IMU_ONLY || IMU_AND_SPEECH


#if OLED_ON

    //u8x8.setI2CAddress(0x78);  //0x3C, address may not be a problem. 
    u8x8.begin();
    pre(); // print the top line -> this freezes when tflite invoke() is called. 0418. 
    


#endif


}

int n=0;
// Handler 
void onRxCharValueUpdate(BLEDevice central, BLECharacteristic characteristic) {
  // central wrote new value to characteristic, update LED
  Serial.print("Characteristic event, read: ");
  byte tmp[512];
  int dataLength = rxChar.readValue(tmp, 512);

  for(int i = 0; i < dataLength; i++) {
    Serial.print((char)tmp[i]);
  }
  Serial.println();
  Serial.print("Value length = ");
  Serial.println(rxChar.valueLength());
  

  last_rXTime_ms = millis(); 
  displayCurrent(rxChar.valueLength());
  
}


uint16_t currentStep = 0;
uint16_t prevStep = 0;
float roll, pitch; //  tilt angle
float yaw; 


// Global variables for previous accelerometer readings
float rollAcc = 0.0;
float pitchAcc = 0.0;											  

static unsigned long last_steptime_ms = 0; // for timer in step count reporting. 
 
 // The name of this function is important for Arduino compatibility.
void loop() {


    /*----- Free fall detection: if free fall is detected, reporting this is a priority. ---------*/
    uint8_t readDataByte = 0;
          // pedometer reading 
    uint8_t readDataByte_step1 = 0;
    uint8_t readDataByte_step2 = 0;
    String buff;
    
    //Read the wake-up source register
    myIMU.readRegister(&readDataByte, LSM6DS3_ACC_GYRO_WAKE_UP_SRC);
    //Mask off the FF_IA bit for free-fall detection
    readDataByte &= 0x20;
    if (readDataByte) {
        detectCount ++;
        Serial.print("Free fall detected!  ");
        Serial.println(detectCount);
        
        buff += String('a'); // a: freefall, b: activity, c: step count report
        buff += F(",");
        buff += String(0); //  If freefall, we really don't care about the following numbers...
        buff += F(",");
        buff += String(0);
        buff += F(",");
        buff += String(0);
        buff += F(",");
        buff += String(0);
        buff += F(",");
        buff += String(0);
//        buff += F(",");
//        buff += String(0);
//        buff += F(",");
//        buff += String(0);
        Serial.println(buff);
        is_free_fall_detected = true;

    //    buff += F(",");
    }// end of free fall detection. 

	  else {
      //TODO: steps will need to be polled at every 10 seconds. 

        
        if (millis() > last_steptime_ms + STEP_REPORT_INTERVAL_MS) {
            last_steptime_ms = millis();

            myIMU.readRegister(&readDataByte_step1, LSM6DS3_ACC_GYRO_STEP_COUNTER_H);
            myIMU.readRegister(&readDataByte_step2, LSM6DS3_ACC_GYRO_STEP_COUNTER_L);
      	
      	    // Combine the two 8-bit integers to form a 16-bit integer
            currentStep = (uint16_t)readDataByte_step1 << 8 | readDataByte_step2;
            
		        buff += String('c');// a: freefall, b: activity, c: step count report
            buff += F(",");
            buff += String(currentStep);// no decoding index necessary 
            buff += F(",");

            Serial.print("Pedometer step:  ");
            Serial.println(currentStep);
            displayStep(currentStep); 
                       
            Serial.println(buff);
            is_stepcount_report = true;
        }


#if IMU_ONLY || IMU_AND_SPEECH
//          int16_t aX, aY, aZ, gX, gY, gZ; // -32767 to 32768
//          // read the acceleration data
//          aX = myIMU.readRawAccelX();
//          aY = myIMU.readRawAccelY();
//          aZ = myIMU.readRawAccelZ();
//          
//          // sum up the absolutes
//          //int16_t aSum = fabs(aX) + fabs(aY) + fabs(aZ)
//          uint16_t aSum = (uint16_t) abs(aX) + (uint16_t) abs(aY) + (uint16_t) abs(aZ);


#if READ_RAW_SIGNAL
        int16_t aX, aY, aZ; 

        aX = myIMU.readRawAccelX();
        aY = myIMU.readRawAccelY();
        aZ = myIMU.readRawAccelZ();
        
        uint16_t aSum = (uint16_t) abs(aX) + (uint16_t) abs(aY) + (uint16_t) abs(aZ);
        
        // check if it's above the threshold
        if (aSum >= accelerationThreshold) {
            // reset the sample read count
            samplesRead = 0;
            Serial.println("Starting inferencing shortly...\n");        
#if PAUSE_AFTER_CUE 
            delay(1000); // prepare the move for 1 second
#endif 
            Serial.println("Sampling...\n");
            //break;
        }

#else  // READ FLOAT
        float aX, aY, aZ, gX, gY, gZ; // int16_t: -32767 to 32768
        // read the acceleration data
        aX = myIMU.readFloatAccelX();
        aY = myIMU.readFloatAccelY();
        aZ = myIMU.readFloatAccelZ();
        
        // sum up the absolutes
        float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
    
//        Serial.print("aSum: ");
//        Serial.print(aSum);
//        Serial.print(" | accelerationThreshold: ");
//        Serial.println(accelerationThreshold);
  	  
        // check if it's above the threshold
        if (aSum >= accelerationThreshold) {
          	// reset the sample read count
          	// wait for significant motion
          	samplesRead = 0;
          	
          	// 20240826:  new temporal setting- heeltap movement is a trigger in initiate inference. 
          	// after 500ms, sampling will happen and the result will be regarded as a new movement or status.
            Serial.println("Starting inferencing shortly...\n");        
#if PAUSE_AFTER_CUE 
          	delay(1000);
#endif 
            Serial.println("Sampling...\n");

        }
#endif  // READ_RAW_SIGNAL

    		// check if the all the required samples have been read since
    		// the last time the significant motion was detected
    
    		while (samplesRead < numSamples) {
    			// check if new acceleration AND gyroscope data is available
    			  // read the acceleration and gyroscope data
#if READ_RAW_SIGNAL
#if READ_3DOF_ONLY
            int16_t aX, aY, aZ; 

            aX = myIMU.readRawAccelX();
            aY = myIMU.readRawAccelY();
            aZ = myIMU.readRawAccelZ();
            int input_dim = 3;
            
            model_input4imu->data.f[samplesRead * input_dim + 0] = (aX + 32768) / 65536.0;  // e.g. 0,1,2,3,4,5,6,7,8....  like 1D array
            model_input4imu->data.f[samplesRead * input_dim + 1] = (aY + 32768) / 65536.0; 
            model_input4imu->data.f[samplesRead * input_dim + 2] = (aZ + 32768) / 65536.0; 
            
#else
            int16_t aX, aY, aZ, gX, gY, gZ; // signed value

    			  aX = myIMU.readRawAccelX();
    			  aY = myIMU.readRawAccelY();
    			  aZ = myIMU.readRawAccelZ();
    
    			  gX = myIMU.readRawGyroX();
    			  gY = myIMU.readRawGyroY();
    			  gZ = myIMU.readRawGyroZ();
    
    			  // normalize the IMU data between 0 to 1 and store in the model's
    			  // input tensor
    			  int input_dim = 6;
    			  model_input4imu->data.f[samplesRead * input_dim + 0] = (aX + 32768) / 65536.0;  // e.g. 0,1,2,3,4,5,6,7,8....  like 1D array
    			  model_input4imu->data.f[samplesRead * input_dim + 1] = (aY + 32768) / 65536.0; 
    			  model_input4imu->data.f[samplesRead * input_dim + 2] = (aZ + 32768) / 65536.0; 
    			  model_input4imu->data.f[samplesRead * input_dim + 3] = (gX + 32768) / 65536.0; 
    			  model_input4imu->data.f[samplesRead * input_dim + 4] = (gY + 32768) / 65536.0; 
    			  model_input4imu->data.f[samplesRead * input_dim + 5] = (gZ + 32768) / 65536.0; 

//            Serial.print((aX + 32768) / 65536.0);
//            Serial.print(",");
//            Serial.print((aY + 32768) / 65536.0);
//            Serial.print(",");
//            Serial.print((aZ + 32768) / 65536.0);
//            Serial.print(",");
//            Serial.print((gX + 32768) / 65536.0);
//            Serial.print(",");
//            Serial.print((gY + 32768) / 65536.0);
//            Serial.print(",");
//            Serial.print((gZ + 32768) / 65536.0);
//            Serial.println(",");
//         //Serial.println("Just before Invoking.");
//            Serial.print("samplesRead: ");
//            Serial.println(samplesRead);

#endif // READ_3DOF_ONLY

#else //READ FLOAT

#if READ_3DOF_ONLY
            float aX, aY, aZ;

            aX = myIMU.readFloatAccelX();
            aY = myIMU.readFloatAccelY();
            aZ = myIMU.readFloatAccelZ();
            int input_dim = 3;

            model_input4imu->data.f[samplesRead * input_dim + 0] = (aX + 32) / 64.0;  // e.g. 0,1,2,3,4,5,6,7,8....  like 1D array
            model_input4imu->data.f[samplesRead * input_dim + 1] = (aY + 32) / 64.0; 
            model_input4imu->data.f[samplesRead * input_dim + 2] = (aZ + 32) / 64.0; 
#else

            float aX, aY, aZ, gX, gY, gZ; // signed value

            aX = myIMU.readFloatAccelX();
            aY = myIMU.readFloatAccelY();
            aZ = myIMU.readFloatAccelZ();
    
            gX = myIMU.readFloatGyroX();
            gY = myIMU.readFloatGyroY();
            gZ = myIMU.readFloatGyroZ();
    
            // normalize the IMU data between 0 to 1 and store in the model's
            // input tensor
            int input_dim = 6;
//            model_input4imu->data.f[samplesRead * input_dim + 0] = (aX + 4) / 8.0;  // e.g. 0,1,2,3,4,5,6,7,8....  like 1D array
//            model_input4imu->data.f[samplesRead * input_dim + 1] = (aY + 4) / 8.0; 
//            model_input4imu->data.f[samplesRead * input_dim + 2] = (aZ + 4) / 8.0; 
//            model_input4imu->data.f[samplesRead * input_dim + 3] = (gX + 2000) / 8000.0; 
//            model_input4imu->data.f[samplesRead * input_dim + 4] = (gY + 2000) / 8000.0; 
//            model_input4imu->data.f[samplesRead * input_dim + 5] = (gZ + 2000) / 8000.0; 

            model_input4imu->data.f[samplesRead * input_dim + 0] = (aX + 32) / 64.0;  // e.g. 0,1,2,3,4,5,6,7,8....  like 1D array
            model_input4imu->data.f[samplesRead * input_dim + 1] = (aY + 32) / 64.0; 
            model_input4imu->data.f[samplesRead * input_dim + 2] = (aZ + 32) / 64.0; 
            model_input4imu->data.f[samplesRead * input_dim + 3] = (gX + 2000) / 4000.0; 
            model_input4imu->data.f[samplesRead * input_dim + 4] = (gY + 2000) / 4000.0; 
            model_input4imu->data.f[samplesRead * input_dim + 5] = (gZ + 2000) / 4000.0; 
#endif  // READ_3DOF_ONLY

#endif  // READ_RAW_SIGNAL
    			  samplesRead++;

    			 if (samplesRead == numSamples) {
      				//Serial.println("in if");
      
      				// Run the model on the spectrogram input and make sure it succeeds.
      				TfLiteStatus invoke_status = interpreter_imu->Invoke();
      				if (invoke_status != kTfLiteOk) {
      					  Serial.println("Invoke failed!");
      					  while (1);         
      					  return;
      				}
      
      				float scores[NUM_GESTURES]={};  // initialize to zeros
      
      				// modified to custom output for performance analysis
      				for (int i = 0; i < NUM_GESTURES; i++) {
//      					  Serial.print(GESTURES[i]);
//      					  Serial.print(": ");
//      					  
//      					  Serial.print(model_output4imu->data.f[i], 6);
//      					  Serial.println(",");			  
      					  scores[i] = model_output4imu->data.f[i]; // costly?
      				}
      
      				  // get gesture with max probability 
      				// get the index of the max in the array. 
      				// start of the task DBVDS
      				int k = 0;
      				float maxscore = scores[k];
      
      				for (int i = 0; i < NUM_GESTURES ; ++i)
      				{
        					if (scores[i] > maxscore)
        					{
          						maxscore = scores[i];
          						k = i;
        					}
      				}
      				decoded_state_index = k;
      				is_new_gesture = true;
       
           }//if
        }//while

#endif //IMU_ONLY || IMU_AND_SPEECH


/*------calculate tilt angle once per inference -------*/ 

		// Assuming 16-bit raw data, where 32768 represents 0 g
    //    float faX = (aX - 32768) / 32768.0;
    //    float faY = (aY - 32768) / 32768.0;
    //    float faZ = (aX - 32768) / 32768.0;
//        float faX = myIMU.readFloatAccelX();
//        float faY = myIMU.readFloatAccelY();
//        float faZ = myIMU.readFloatAccelZ();
//        
//        // Compute roll and pitch from accelerometer
//        rollAcc = atan2(faY, faZ ) * 180 / PI;
//        pitchAcc  = atan2(-1*faX, sqrt(faY * faY + faZ * faZ)) * 180 / PI;

//		// Print the results
//		// combination sleeping. (supine + prone + side?)
//		  // Print the results
//		Serial.print("Roll: ");
//		Serial.print(rollAcc); // Convert to degrees
//		Serial.print(" | Pitch: ");
//		Serial.println(pitchAcc); // Convert to degrees

        if (is_new_gesture) {
            //String buff;
            //  buff += String(aX, 2);
            //  buff += F(",");
            //  buff += String(aY, 2);
            //  buff += F(",");
            //  buff += String(aZ, 2);
            //  buff += F(",");
            buff += String('b');// a: freefall, b: activity, c: step count report
            buff += F(",");
            buff += String(decoded_state_index);
            buff += F(",");
            buff += String(model_output4imu->data.f[0]);
            buff += F(",");
            buff += String(model_output4imu->data.f[1]);
            buff += F(",");
            buff += String(model_output4imu->data.f[2]);
            buff += F(",");
            buff += String(model_output4imu->data.f[3]);
//            buff += F(",");
//            buff += String(model_output4imu->data.f[4]);
            
//            buff += F(",");
//            buff += String(model_output4imu->data.f[5]);
            Serial.println(GESTURES[decoded_state_index]);
            Serial.println(buff);
        }

	  } // else (if not free-fall). 
    // either FREE-FALL or 

#if BLE_ON
    // BLE
    // wait for a BLE central
    central = BLE.central();
    
    // if a central is connected to the peripheral:
    if (central) {
#if PRINTOUT_ABOVE_SERIALOUT_PLOTTING
      	Serial.print("Connected to central: ");
      	// print the central's BT address:
      	Serial.println(central.address());
#endif
      	// turn on the LED to indicate the connection:
      	//digitalWrite(LED_BUILTIN, HIGH);


  			// check the IMU every period
  			// while the central is connected:
  			if (central.connected()) {  // instead of while loop use if for a single pass. 
  				  //currentMillis = micros();
  				  /* Check if the time in between samples equals sampling period before taking a new sample */
  				  // if (deltaTime >= period) {
  				  //if ( currentMillis - previousMillis > 500) { // 500ms for bluetooth sending inverval
  				  if (is_free_fall_detected | is_new_gesture | is_stepcount_report ){ // 20230512 UPDATE
  					/* Display the current time in microseconds */
#if PRINTOUT_ABOVE_SERIALOUT_PLOTTING
      					Serial.print(currentTime);
      					Serial.print(",");
#endif
                Serial.print("in BLE: ");
                Serial.println(buff);
  				        //previousMillis = currentMillis;
  
  
          		// Stores each IMU data type in a string 'buff' //
          		//original
          //        String buff;
          //        buff += String(currentTime);
          //        buff += F(",");
          //        buff += String(aX, 2);
          //        buff += F(",");
          //        buff += String(aY, 2);
          //        buff += F(",");
          //        buff += String(aZ, 2);
          //        buff += F(",");
          //        buff += String(deltaTime);
          //        buff += F(",");
          		// Sends buff using the BLE connection


          //        Serial.print(String(aX, 2));
          //        Serial.print(",");
          //        Serial.print(String(aY, 2));
          //        Serial.print(",");
          //        Serial.print(String(aZ, 2));
          //        Serial.print(",");
          //        Serial.print(model_output4imu->data.f[0]);
          //        Serial.print(",");
          //        Serial.print(model_output4imu->data.f[1]);
          //        Serial.print(",");
          //        
          //       
          //        Serial.println(buff);
          //        
#if PRINTOUT_ABOVE_SERIALOUT_PLOTTING 
			          Serial.print("Sent to Central: ");
			          // Serial.println(buff);
      
			          Serial.println(GESTURES[decoded_state_index]);
#endif
					        //IMUSensorData.writeValue(GESTURES[decoded_state_index]);

#if IMU_ONLY 

                if (is_free_fall_detected) {
                    Serial.print("Free-fall buff: ");
                    Serial.println(buff);
                    IMUSensorData.writeValue(buff);
                }
      					if (is_new_gesture) {
        						Serial.print("Activity buff: ");
        						Serial.println(buff);
        						IMUSensorData.writeValue(buff);
      					} 
                  // TODO 
                if (is_stepcount_report) {
                    Serial.print("Step Count buff: ");
                    Serial.println(buff);
                    IMUSensorData.writeValue(buff);
                } 
#endif  

	        	}
      
        }
  
      
  			// when the central disconnects, turn off the LED:
  			//digitalWrite(LED_BUILTIN, LOW);
       
  			//Serial.print("Disconnected from central: ");
  			//Serial.println(central.address());
		}
#endif // BLE_ON




      
#if OLED_ON
        
		/* if (millis() > last_idle_start_time_ms + OLED_IDLE_TURNOFF_MS ) {
			u8x8.clearLine(0); // clear row 0
			u8x8.clearLine(3); // clear row 4
			u8x8.clearLine(4); // clear row 5
		}

		if (is_new_speech_command | is_new_gesture ) {
			is_new_gesture = false;
			last_idle_start_time_ms = millis();
			u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);


#if IMU_ONLY
			gesture_header();
			u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
			u8x8.drawString(0, 3, GESTURES[decoded_state_index]);
		} 
#endif

		*/
												   

  		if (is_free_fall_detected) {
    			// Quick refresh of screen when free fall is detected
          digitalWrite(RED_ledPin, LOW);
          digitalWrite(BLUE_ledPin, HIGH);                          // freefall       red
          digitalWrite(GREEN_ledPin, HIGH);
    			u8x8.clearLine(0); // clear row 0
    			u8x8.clearLine(3); // clear row 4
    			u8x8.clearLine(4); // clear row 5
    			activity_header();
  
  				
  							 
    			u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
    			u8x8.drawString(0, 3,"FREEFALL");
    			is_free_fall_detected = false;
#if INSERT_DELAY_AFTER_FREEFALL
          delay(120); 
#endif
  		}else if( is_new_gesture ) { 
          digitalWrite(RED_ledPin, HIGH);
          digitalWrite(BLUE_ledPin, HIGH);                         
          digitalWrite(GREEN_ledPin, LOW);        // activity     green
    			is_new_gesture = false;
    			last_idle_start_time_ms = millis();
    			u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);
    			
    			u8x8.clearLine(0); // clear row 0
    			u8x8.clearLine(3); // clear row 4
    			u8x8.clearLine(4); // clear row 5
    			activity_header();
    			u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
    			u8x8.drawString(0, 3, GESTURES[decoded_state_index]);
  		}
      else if (is_stepcount_report)  {
          is_stepcount_report = false;
          digitalWrite(RED_ledPin, HIGH);
          digitalWrite(BLUE_ledPin, LOW);           // step count report blue               
          digitalWrite(GREEN_ledPin, HIGH);       
     
  		}
  		  
  	//    if (millis() > last_rXTime_ms + 4000) { // display call back current value for 4s
  	//       u8x8.clearLine(6); // clear row 6
  	//       u8x8.clearLine(7); // clear row 7
  	//    }
  
  		if (millis() > last_stepTime_ms + 2000) { // display call back current value for 2s
  		   u8x8.clearLine(6); // clear row 6
  		   u8x8.clearLine(7); // clear row 7
  		}
		

	/* 	if (millis() > last_rXTime_ms + 4000) { // display call back current value for 4s
			u8x8.clearLine(6); // clear row 6
			u8x8.clearLine(7); // clear row 7
		} */
    
#endif // OLED_ON




}

/* OLED display related */
void pre(void)
{
	u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);    
	u8x8.clear();

	u8x8.inverse();
	u8x8.print(" Kinadaptive SCS");
	u8x8.setFont(u8x8_font_chroma48medium8_r);  
	u8x8.noInverse();
	u8x8.setCursor(0,1);
#if BLE_ON
	//  u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);
	//  u8x8.print(" Sending to IPG");
	u8x8.setFont(u8x8_font_open_iconic_embedded_2x2); //https://github.com/olikraus/u8g2/wiki/fntgrpiconic#open_iconic_embedded_8x
	u8x8.drawGlyph(30,1, '@'+10); // bluetooth icon?
	//
	//  u8x8.setFont(u8x8_font_chroma48medium8_r);  
	//  u8x8.setCursor(0,2);
#endif

}



/* OLED display related */
void activity_header(void)
{
	u8x8.clearLine(0); // clear row 0
	u8x8.setCursor(0,0);
	u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);    
	u8x8.inverse();
	u8x8.print("  ACTIVITY   ");
	u8x8.setFont(u8x8_font_amstrad_cpc_extended_f);  
	u8x8.noInverse();
	u8x8.setCursor(0,1);
}


/* OLED display related */
void clearMainMessage(void)
{
	//su8x8.clearLine(3); // clear row 3
	u8x8.clearLine(3); // clear row 4
	u8x8.clearLine(4); // clear row 5
	u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
}


/* OLED display related */
void displayStep(int curStep)
{
	u8x8.clearLine(6); // clear row 6
	u8x8.clearLine(7); // clear row 7
	u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
	u8x8.inverse();
	u8x8.setCursor(0,6);  // col, row
	u8x8.print(String(curStep) + " stps");
	u8x8.setCursor(6,6);  // col, row
	u8x8.setFont(u8x8_font_open_iconic_arrow_2x2);
	u8x8.noInverse();
	u8x8.setCursor(1,6);
	 
}



/* OLED display related */
void displayCurrent(int I)
{
	float f1 = (float)I/10;
	u8x8.clearLine(6); // clear row 6
	u8x8.clearLine(7); // clear row 7
	u8x8.setFont(u8x8_font_px437wyse700b_2x2_r);
	u8x8.inverse();
	u8x8.setCursor(0,6);  // col, row
	u8x8.print(String(f1) + "mA");
	u8x8.setCursor(6,6);  // col, row
	//u8x8.print(String(f1));
	u8x8.setFont(u8x8_font_open_iconic_arrow_2x2);
	if (decoded_state_index == 0) {    
		u8x8.drawGlyph(12,6, '@'+11); // arrow up icon
	}else if (decoded_state_index ==1 ) { 
		u8x8.drawGlyph(12,6, '@'+8); // arrow down icon
	}
	  
	//u8x8.setCursor(2,7);
	//u8x8.print(" mA");
	u8x8.noInverse();
	u8x8.setCursor(1,6);

}



void config_free_fall_detect(void) {
	uint8_t error = 0;
	uint8_t dataToWrite = 0;
  uint8_t dataToWriteGyro = 0;

	dataToWrite |= LSM6DS3_ACC_GYRO_BW_XL_100Hz;
	dataToWrite |= LSM6DS3_ACC_GYRO_FS_XL_2g;
	dataToWrite |= LSM6DS3_ACC_GYRO_ODR_XL_104Hz;

	//The FF_DUR[5:0] field of the FREE_FALL / WAKE_UP_DUR registers is configured like this
	//to ignore events that are shorter than 6/ODR_XL = 6/412 Hz ~= 15 msec in order to avoid false detections.
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_CTRL1_XL, dataToWrite); // dataToWrite doesn't work. 0x60 works. HMM?
	
	
	dataToWriteGyro |= LSM6DS3_ACC_GYRO_ODR_G_104Hz;
  dataToWriteGyro |= LSM6DS3_ACC_GYRO_FS_G_2000dps;
  
  error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_CTRL2_G, dataToWriteGyro); // dataToWrite doesn't work. 0x60 works. HMM?


	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_WAKE_UP_DUR, 0x00); // set event duration (FF_DUR5 bit)
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_FREE_FALL, 0x33); // 0x33: 312 mg for free-fall recognition.  free fall threshold (FF_THS[2:10]= 011b), // Set six samples event duration (FF_DUR[5:0] = 000110b)
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_MD1_CFG, 0x10);  // FF interrupt driven to INT1 pin
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_MD2_CFG, 0x10);

	// pedometer specific config
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_CTRL10_C, 0x3E); // Pedometer X, Y, Z axis output enabled,  reset pedo stepc counter.
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_INT1_CTRL, 0x80); // Step detector interrupt driven to INT1 pin
	error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_CONFIG_PEDO_THS_MIN, 0x80); // 


  error += myIMU.writeRegister(LSM6DS3_ACC_GYRO_TAP_CFG1, 0x81);// 0x01 is an error,  enable interrupts and latch interrupt)




	if (error) {
		Serial.println("Error during free fall config.");
	}else  {
		Serial.println("Free-fall config success!");
	}
}


//
//void int1ISR()
//{
//	//Serial.println("Int1ISR"); /// Any serial line print causes freeze
//	interruptCount++;
//
//}
