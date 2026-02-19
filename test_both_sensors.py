#!/usr/bin/env python3
"""
Combined MPU6050 + MAX30102 Sensor Test
Tests both sensors running simultaneously from the same I2C bus
"""

import sys
import time
import logging
from uuid import uuid4

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, '/home/hero/hero_system/hero_system')
sys.path.insert(0, '/home/hero/hero_core')
sys.path.insert(0, '/home/hero/hardware-calibration/tests/Accel_Oxi_TB/max30102-master')

def test_sensors():
    """Test both sensors simultaneously"""
    
    print("=" * 70)
    print("HERO System - Combined Sensor Test")
    print("MPU6050 (Accelerometer + Gyroscope) + MAX30102 (Heart Rate + SpO2)")
    print("=" * 70)
    
    try:
        # Import modules
        from hero_system.coordinator import SensorCoordinator
        from hero_system.sensors.mpu6050 import MPU6050Collector, MPU6050Config
        from hero_system.sensors.max30102 import MAX30102Collector, MAX30102Config
        
        print("\n✓ Modules imported successfully")
        
        # Create mock database session (for testing without DB)
        class MockDBSession:
            def add(self, obj):
                pass
            def commit(self):
                pass
            def rollback(self):
                pass
        
        db_session = MockDBSession()
        session_id = uuid4()
        
        print(f"✓ Test session ID: {session_id}")
        
        # Initialize coordinator
        print("\n1. Initializing coordinator...")
        coordinator = SensorCoordinator()
        coordinator.start()
        print("✓ Coordinator started")
        
        # Configure sensors for calibration mode (real-time display)
        mpu_config = MPU6050Config.for_calibration()
        max_config = MAX30102Config.for_calibration()
        
        # Initialize collectors
        print("\n2. Initializing MPU6050 (Accelerometer + Gyroscope)...")
        mpu_collector = MPU6050Collector(
            session_id=session_id,
            db_session=db_session,
            coordinator=coordinator,
            config=mpu_config
        )
        
        print("\n3. Initializing MAX30102 (Heart Rate + SpO2)...")
        max_collector = MAX30102Collector(
            session_id=session_id,
            db_session=db_session,
            coordinator=coordinator,
            config=max_config
        )
        
        # Start both collectors
        print("\n4. Starting data collection...")
        print("-" * 70)
        mpu_collector.start()
        time.sleep(0.5)
        max_collector.start()
        
        print("\n✓ Both sensors collecting data!")
        print("\nPlace finger on MAX30102 sensor...")
        print("Move/shake device to test MPU6050...")
        print("\nPress Ctrl+C to stop\n")
        print("-" * 70)
        
        # Monitor data collection
        start_time = time.time()
        last_status = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Print status every 5 seconds
            if current_time - last_status >= 5.0:
                print(f"\n[{elapsed:.1f}s] Status:")
                print(f"  MPU6050  - Accel samples: {mpu_collector.accel_sample_count}, "
                      f"Gyro samples: {mpu_collector.gyro_sample_count}")
                print(f"  MAX30102 - Samples: {max_collector.sample_count}")
                
                # Show latest values if in calibration mode
                if hasattr(mpu_collector, 'latest_accel_magnitude'):
                    print(f"  MPU6050  - Latest accel magnitude: {mpu_collector.latest_accel_magnitude:.2f} m/s²")
                
                if hasattr(max_collector, 'latest_hr') and max_collector.latest_hr_valid:
                    hr = max_collector.latest_hr
                    if max_collector.latest_spo2_valid:
                        spo2 = max_collector.latest_spo2
                        print(f"  MAX30102 - HR: {hr} bpm, SpO2: {spo2}%")
                    else:
                        print(f"  MAX30102 - HR: {hr} bpm, SpO2: ---")
                
                last_status = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Stopping sensors...")
        print("=" * 70)
        
        try:
            max_collector.stop()
            print("✓ MAX30102 stopped")
        except Exception as e:
            print(f"✗ Error stopping MAX30102: {e}")
        
        try:
            mpu_collector.stop()
            print("✓ MPU6050 stopped")
        except Exception as e:
            print(f"✗ Error stopping MPU6050: {e}")
        
        try:
            coordinator.stop()
            print("✓ Coordinator stopped")
        except Exception as e:
            print(f"✗ Error stopping coordinator: {e}")
        
        print("\n" + "=" * 70)
        print("Test completed!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    test_sensors()
    
