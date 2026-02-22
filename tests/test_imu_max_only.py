#!/usr/bin/env python3
"""
Direct test of MPU6050 + MAX30102 sensors
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
    print("HERO System - IMU + MAX30102 Test")
    print("=" * 70)
    
    try:
        # Import modules
        from hero_system.coordinator.coordinator import SensorCoordinator
        from hero_system.sensors.mpu6050.collector import MPU6050Collector
        from hero_system.sensors.mpu6050.config import MPU6050Config
        from hero_system.sensors.max30102.collector import MAX30102Collector
        from hero_system.sensors.max30102.config import MAX30102Config
        
        print("\n✓ Modules imported successfully")
        
        # Create mock database session
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
        
        # Initialize coordinator (just for timestamps)
        print("\n1. Initializing coordinator...")
        coordinator = SensorCoordinator(session_id=session_id, db_session=db_session)
        print("✓ Coordinator ready")
        
        # Configure sensors for calibration mode
        mpu_config = MPU6050Config.for_calibration()
        max_config = MAX30102Config.for_calibration()
        
        # Initialize collectors
        print("\n2. Initializing MPU6050...")
        mpu_collector = MPU6050Collector(
            session_id=session_id,
            db_session=db_session,
            coordinator=coordinator,
            config=mpu_config
        )
        
        print("\n3. Initializing MAX30102...")
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
        
        print("\n✓ Both sensors running!")
        print("\nPlace finger on MAX30102...")
        print("Move device to test MPU6050...")
        print("\nPress Ctrl+C to stop\n")
        
        # Monitor
        start_time = time.time()
        last_status = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_status >= 5.0:
                print(f"\n[{elapsed:.1f}s] Status:")
                print(f"  MPU6050  - Accel: {mpu_collector.accel_sample_count}, Gyro: {mpu_collector.gyro_sample_count}")
                print(f"  MAX30102 - Samples: {max_collector.sample_count}")
                last_status = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        try:
            max_collector.stop()
            print("✓ MAX30102 stopped")
        except:
            pass
        try:
            mpu_collector.stop()
            print("✓ MPU6050 stopped")
        except:
            pass
        print("✓ Test completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    test_sensors()
    
